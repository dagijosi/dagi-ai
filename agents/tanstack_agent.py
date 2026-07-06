import re
from typing import Dict, Optional, List, Tuple
from api.lmstudio_client import call_llm
from tools.tool_manager import ToolManager


class TanStackAgent:
    """Specialized agent for TanStack Query (v5+) & TypeScript generation tasks.
    
    This agent is interactive - it:
    - Accepts a target directory path to create/modify files
    - Reads existing files in the target for context
    - Creates directory structures and writes files to disk
    - Verifies files by reading them back after creation
    - Has persistent memory: stores past generations and learns from corrections
    """
    
    # Regex to extract file blocks from LLM output: ```filepath:path/to/file.ext ... ```
    _FILE_BLOCK_RE = re.compile(
        r'```(?:filepath:|file:)?([^\n]+?)\n(.*?)```',
        re.DOTALL
    )
    # Fallback: match any markdown code block with a file-like name
    _CODE_BLOCK_RE = re.compile(
        r'```(\w*)\n(.*?)```',
        re.DOTALL
    )
    
    def __init__(self, tool_manager: Optional[ToolManager] = None, memory_agent=None):
        self.status = "idle"
        self.tool_manager = tool_manager
        self.memory_agent = memory_agent
        self._ensure_memory_agent()
        self.system_prompt = """You are a TanStack Query (v5+) & TypeScript Generation Agent. Generate code following this architecture:

ARCHITECTURE: src/connections/{Module}/function.ts + index.ts (PascalCase folders)
- function.ts: Pure API calls via client.get<T>()/post<T>() etc. NO @tanstack/react-query imports
- index.ts: TanStack Query hooks using ./function. Import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
- client.ts: SHARED client at src/connections/client.ts (common for all modules) with .get<T>(), .post<T>(), .put<T>(), .patch<T>(), .delete<T>()
- Types: In src/types/ folder
- Query keys: Centralized in src/constants/queryKeys.ts with pattern keys.module.list(filters), .lists(), .detail(id)
- Error: CustomError class with statusCode
- Toast: premiumToast from ../../components/ui/feedback
- Store: Zustand stores from ../../store/
- QueryClientProvider: In main app file (not separate provider)

IMPORTANT: client.ts is a SHARED file at src/connections/client.ts. All modules import from this common client.
Import path: import { client } from "../client" (from function.ts in src/connections/{Module}/)

Hook naming: useXQuery (queries), useXMutation (mutations)
Mutation pattern: onSuccess invalidates queries + premiumToast.success, onError premiumToast.error

OUTPUT FORMAT: Use ```filepath:path/to/file\ncode``` blocks. Each block = one file.""" 
    
    def execute(self, task_data: Dict) -> Dict:
        """Execute a TanStack Query task."""
        self.status = "working"
        task_type = task_data.get('type')
        
        try:
            if task_type == 'initialize':
                result = self._initialize_project(task_data)
            elif task_type == 'ingest_api':
                result = self._ingest_api_endpoint(task_data)
            elif task_type == 'generate_hook':
                result = self._generate_hook(task_data)
            elif task_type == 'generate_types':
                result = self._generate_types(task_data)
            elif task_type == 'generate_functions':
                result = self._generate_functions(task_data)
            elif task_type == 'refactor':
                result = self._refactor_tanstack(task_data)
            elif task_type == 'debug':
                result = self._debug_tanstack(task_data)
            elif task_type == 'explain':
                result = self._explain_tanstack(task_data)
            else:
                # Default to project initialization or API ingestion based on context
                if task_data.get('api_endpoint') or task_data.get('payload'):
                    result = self._ingest_api_endpoint(task_data)
                else:
                    result = self._initialize_project(task_data)
            
            self.status = "idle"
            return result
        except Exception as e:
            self.status = "error"
            return {'error': str(e), 'status': 'error'}
    
    # ── File system helpers ──────────────────────────────────────────
    
    def _get_file_tool(self):
        """Get the file tool from tool_manager or create a fallback."""
        if self.tool_manager:
            return self.tool_manager.get_tool('file')
        from tools.file_tool import FileTool
        return FileTool()
    
    def _resolve_path(self, target_path: str, sub_path: str = "") -> str:
        """Resolve a file path relative to the target directory."""
        # Clean and normalize the path
        target = target_path.replace('\\', '/').rstrip('/')
        if sub_path:
            sub = sub_path.replace('\\', '/').lstrip('/')
            return f"{target}/{sub}"
        return target
    
    def _read_file_content(self, file_path: str) -> Optional[str]:
        """Read file content using the file tool."""
        try:
            file_tool = self._get_file_tool()
            result = file_tool.execute(operation='read', file_path=file_path)
            if result.get('status') == 'read' and 'content' in result:
                return result['content']
        except Exception:
            pass
        return None
    
    def _check_directory_exists(self, dir_path: str) -> bool:
        """Check if a directory exists."""
        try:
            file_tool = self._get_file_tool()
            result = file_tool.execute(operation='list', dir_path=dir_path)
            return result.get('status') == 'listed'
        except Exception:
            return False
    
    def _ensure_directory(self, dir_path: str) -> bool:
        """Ensure a directory exists, creating it if needed."""
        try:
            file_tool = self._get_file_tool()
            # Try to list it first - if it fails, create a placeholder
            check = file_tool.execute(operation='list', dir_path=dir_path)
            if check.get('status') == 'listed':
                return True
            # Write a .gitkeep to create the directory
            keep_path = self._resolve_path(dir_path, '.gitkeep')
            file_tool.execute(operation='write', file_path=keep_path, content='')
            return True
        except Exception as e:
            print(f"Warning: Could not ensure directory {dir_path}: {e}")
            return False
    
    def _write_file(self, file_path: str, content: str) -> Dict:
        """Write content to a file using the file tool."""
        try:
            file_tool = self._get_file_tool()
            result = file_tool.execute(
                operation='write',
                file_path=file_path,
                content=content
            )
            return result
        except Exception as e:
            return {'error': str(e)}
    
    def _verify_file(self, file_path: str) -> Dict:
        """Verify a file was written correctly by reading it back."""
        content = self._read_file_content(file_path)
        if content is None:
            return {'verified': False, 'error': f'Could not read {file_path} back'}
        return {
            'verified': True,
            'file_path': file_path,
            'size': len(content),
            'content_preview': content[:200] + ('...' if len(content) > 200 else '')
        }
    
    # ── LLM output parsing ──────────────────────────────────────────
    
    def _extract_file_blocks(self, llm_output: str) -> List[Tuple[str, str]]:
        """Extract file blocks from LLM output.
        
        Returns list of (file_path, content) tuples.
        Supports formats:
        - ```filepath:path/to/file.ts\ncontent```
        - ```file:path/to/file.ts\ncontent```
        - ```typescript\n// path/to/file.ts\ncontent```  (with path comment)
        """
        blocks = []
        
        # Try primary format: ```filepath:path.ext ... ```
        for match in self._FILE_BLOCK_RE.finditer(llm_output):
            filepath = match.group(1).strip()
            content = match.group(2).strip()
            if filepath and content:
                blocks.append((filepath, content))
        
        # If no blocks found with primary format, try fallback
        if not blocks:
            for match in self._CODE_BLOCK_RE.finditer(llm_output):
                lang = match.group(1).strip().lower()
                content = match.group(2).strip()
                
                # Check first line for a file path comment
                first_line = content.split('\n')[0].strip()
                path_match = re.match(r'^//\s*(.+\.(?:ts|tsx|js|jsx|css|json|md))', first_line)
                if path_match:
                    filepath = path_match.group(1).strip()
                    # Remove the path comment line from content
                    content_lines = content.split('\n')[1:]
                    content = '\n'.join(content_lines).strip()
                    if filepath and content:
                        blocks.append((filepath, content))
                # Also check for ```ts or ```typescript with a sensible name
                elif lang in ('ts', 'typescript', 'tsx', 'js', 'jsx'):
                    # Try to infer filename from context - won't be perfect
                    pass  # Skip fallback for now
        
        return blocks
    
    # ── Memory & Learning ──────────────────────────────────────────

    def _ensure_memory_agent(self):
        """Initialize memory agent if not provided."""
        if self.memory_agent is None:
            try:
                from agents.memory_agent import MemoryAgent
                self.memory_agent = MemoryAgent()
            except Exception:
                self.memory_agent = None
    
    def _ensure_shared_client(self, target_path: str):
        """Ensure the shared client.ts file exists at src/connections/client.ts."""
        client_path = f"{target_path}/src/connections/client.ts"
        
        # Check if client.ts already exists
        existing_content = self._read_file_content(client_path)
        if existing_content:
            return  # Already exists
        
        # Create the shared client.ts
        client_content = """import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';

/**
 * Shared HTTP client for all API connections.
 * Provides typed methods for common HTTP operations with built-in error handling.
 */
class ApiClient {
  private client: AxiosInstance;
  private baseURL: string;

  constructor(baseURL: string = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000') {
    this.baseURL = baseURL;
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor for auth
    this.client.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('authToken');
        if (token && config.headers) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          // Handle unauthorized - redirect to login
          localStorage.removeItem('authToken');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  async get<T>(
    url: string,
    config?: AxiosRequestConfig & { includeAuth?: boolean }
  ): Promise<T> {
    const response = await this.client.get<T>(url, config);
    return response.data;
  }

  async post<T>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig & { includeAuth?: boolean }
  ): Promise<T> {
    const response = await this.client.post<T>(url, data, config);
    return response.data;
  }

  async put<T>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig & { includeAuth?: boolean }
  ): Promise<T> {
    const response = await this.client.put<T>(url, data, config);
    return response.data;
  }

  async patch<T>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig & { includeAuth?: boolean }
  ): Promise<T> {
    const response = await this.client.patch<T>(url, data, config);
    return response.data;
  }

  async delete<T>(
    url: string,
    config?: AxiosRequestConfig & { includeAuth?: boolean }
  ): Promise<T> {
    const response = await this.client.delete<T>(url, config);
    return response.data;
  }
}

// Export singleton instance
export const client = new ApiClient();
"""
        
        # Write the client.ts file
        self._write_file(client_path, client_content)

    def _store_in_memory(self, task_type: str, task_data: Dict, result: Dict):
        """Store a generation result in memory for future learning."""
        if not self.memory_agent:
            return
        try:
            # Extract key info for memory
            task = task_data.get('task', '')
            endpoint = task_data.get('api_endpoint', '')
            method = task_data.get('http_method', '')
            module_name = result.get('module_name', '')
            endpoint_name = result.get('endpoint_name', '')
            files = result.get('files_written', [])
            file_list = ', '.join(f.get('file', '') for f in files if f.get('file'))
            
            # Store API pattern for future reference
            if task_type == 'ingest_api' and endpoint:
                api_pattern = {
                    'endpoint': endpoint,
                    'method': method,
                    'module_name': module_name,
                    'endpoint_name': endpoint_name,
                    'success': result.get('status') == 'ingested'
                }
                
                # Store as structured memory
                content = f"API Pattern: {method} {endpoint}\nModule: {module_name}\nFiles: {file_list}\nStatus: {result.get('status')}"
                
                self.memory_agent.save_memory(
                    content=content,
                    collection='api_patterns',
                    metadata={
                        'agent': 'tanstack',
                        'task_type': task_type,
                        'endpoint': endpoint,
                        'method': method,
                        'module_name': module_name,
                        'endpoint_name': endpoint_name,
                        'files': file_list,
                        'success': result.get('status') == 'ingested',
                        'timestamp': str(__import__('datetime').datetime.now())
                    }
                )
            
            # Also store in general code collection
            content = f"Task: {task}\nEndpoint: {endpoint} {method}\nModule: {module_name}\nFiles: {file_list}"
            
            self.memory_agent.save_memory(
                content=content,
                collection='code',
                metadata={
                    'agent': 'tanstack',
                    'task_type': task_type,
                    'task_description': task,
                    'endpoint': endpoint,
                    'method': method,
                    'module': module_name,
                    'files': file_list
                }
            )
        except Exception:
            pass

    def _retrieve_memory_context(self, task: str, endpoint: str = '') -> str:
        """Retrieve relevant past memory context for a task."""
        if not self.memory_agent:
            return ''
        try:
            memories = []
            
            # First search in api_patterns collection for similar endpoints
            if endpoint:
                api_results = self.memory_agent.search_memory(
                    query=endpoint,
                    collection='api_patterns',
                    n_results=2
                )
                if api_results.get('status') == 'searched' and api_results.get('results'):
                    for i, doc in enumerate(api_results['results']):
                        meta = api_results['metadatas'][i] if i < len(api_results['metadatas']) else {}
                        score = api_results['distances'][i] if i < len(api_results['distances']) else 0
                        if score < 1.0:  # Only include very similar API patterns
                            memories.append(f"- Similar API pattern: {meta.get('method', '')} {meta.get('endpoint', '')} (module: {meta.get('module_name', '')})")
            
            # Also search in general code collection
            query = f"{task} {endpoint}"
            code_results = self.memory_agent.search_memory(
                query=query,
                collection='code',
                n_results=2
            )
            if code_results.get('status') == 'searched' and code_results.get('results'):
                for i, doc in enumerate(code_results['results']):
                    meta = code_results['metadatas'][i] if i < len(code_results['metadatas']) else {}
                    score = code_results['distances'][i] if i < len(code_results['distances']) else 0
                    if score < 1.5:  # Only include relevant memories
                        memories.append(f"- Previous generation: {doc}")
            
            if memories:
                return 'From past work:\n' + '\n'.join(memories) + '\n'
        except Exception:
            pass
        return ''

    def store_correction(self, original: str, correction: str, context: str = '') -> Dict:
        """Store a correction for learning. Call this when user provides feedback.
        
        Args:
            original: The original (incorrect) code or response
            correction: The corrected version
            context: What the task was about
        """
        if not self.memory_agent:
            return {'error': 'Memory agent not available', 'status': 'error'}
        try:
            content = f"Correction for: {context}\nOriginal: {original}\nCorrection: {correction}"
            result = self.memory_agent.save_memory(
                content=content,
                collection='corrections',
                metadata={'agent': 'tanstack', 'context': context}
            )
            return {'status': 'stored', 'result': result}
        except Exception as e:
            return {'error': str(e), 'status': 'error'}

    def _build_context_summary(self, target_path: str, existing_files: Dict[str, str]) -> str:
        """Build a short summary of existing files for the LLM prompt."""
        if not existing_files:
            return "No existing files found in the target directory."
        
        summary_lines = [f"Existing files in {target_path}:"]
        total_chars = 0
        MAX_CHARS = 2000
        
        for filepath, content in existing_files.items():
            if total_chars >= MAX_CHARS:
                summary_lines.append(f"... and {len(existing_files) - len(summary_lines) + 1} more files")
                break
            lines = content.split('\n')
            # Only first 5 lines per file, 200 chars max per file
            snippet = '\n'.join(lines[:5])
            if len(snippet) > 200:
                snippet = snippet[:200] + '...'
            summary_lines.append(f"\n--- {filepath} ---\n{snippet}")
            total_chars += len(snippet)
        
        return ''.join(summary_lines)
    
    def _scan_existing_structure(self, target_path: str) -> Dict[str, str]:
        """Scan the target directory for existing project files.
        
        Returns dict of relative_path -> file_content.
        """
        existing = {}
        
        # Directories to scan
        scan_dirs = [
            target_path,
            self._resolve_path(target_path, 'src'),
            self._resolve_path(target_path, 'src/connections'),
            self._resolve_path(target_path, 'src/constants'),
            self._resolve_path(target_path, 'src/types'),
            self._resolve_path(target_path, 'src/components'),
            self._resolve_path(target_path, 'src/store'),
            self._resolve_path(target_path, 'src/utils'),
            self._resolve_path(target_path, 'app'),
        ]
        
        # File patterns to look for
        important_files = [
            'package.json',
            'tsconfig.json',
            'src/connections/client.ts',
            'src/constants/queryKeys.ts',
            'src/index.tsx',
            'src/App.tsx',
            'src/main.tsx',
            'app/layout.tsx',
            'app/page.tsx',
        ]
        
        # Check important files
        for filepath in important_files:
            full_path = self._resolve_path(target_path, filepath)
            content = self._read_file_content(full_path)
            if content:
                existing[filepath] = content
        
        # Scan known directories for .ts/.tsx files
        for scan_dir in scan_dirs:
            if not self._check_directory_exists(scan_dir):
                continue
            try:
                file_tool = self._get_file_tool()
                result = file_tool.execute(operation='list', dir_path=scan_dir)
                if result.get('status') == 'listed':
                    for item in result.get('items', []):
                        if item['type'] == 'file' and item['name'].endswith(('.ts', '.tsx')):
                            rel_path = self._resolve_path(
                                target_path,
                                f"{scan_dir.replace(target_path, '').lstrip('/')}/{item['name']}"
                            )
                            if rel_path not in existing:
                                content = self._read_file_content(rel_path)
                                if content:
                                    existing[rel_path] = content
            except Exception:
                continue
        
        return existing
    
    def _sanitize_filepath(self, raw_path: str, target_path: str) -> str:
        """Sanitize a file path from the LLM into a clean relative path.
        
        Handles cases where the LLM outputs:
        - `path/to/D:/Projects/x/app/types/index.ts` (path/to/ prefix + abs path)
        - `D:/Projects/x/app/types/index.ts` (full absolute path)
        - `app/types/index.ts` (relative path - correct)
        - `./types/index.ts` (dot-relative path)
        - `path/to/types/index.ts` (path/to/ prefix)
        """
        path = raw_path.replace('\\', '/').strip()
        
        # Remove leading ./ or .\
        if path.startswith('./'):
            path = path[2:]
        
        # Remove leading `path/to/` or `path\to\` that the LLM sometimes adds
        if path.startswith('path/to/'):
            path = path[7:]
        elif path.startswith('path\\to\\'):
            path = path[8:]
        
        # If the path still contains the target_path, extract just the relative part
        target_normalized = target_path.replace('\\', '/').strip('/')
        if target_normalized in path:
            idx = path.find(target_normalized)
            relative = path[idx + len(target_normalized):].lstrip('/')
            if relative:
                return relative
        
        # If it's become an absolute Windows path after stripping, try to make it relative
        # e.g. if `path/to/` is stripped and we're left with `D:/Projects/...`
        if ':' in path or path.startswith('/'):
            # It's still an absolute path, return just the filename as fallback
            return path.split('/')[-1] if '/' in path else path
        
        return path
    
    def _write_generated_files(self, target_path: str, llm_output: str) -> List[Dict]:
        """Write generated files from LLM output to disk.
        
        Extracts file blocks from LLM output, resolves paths, writes files,
        and verifies them.
        """
        blocks = self._extract_file_blocks(llm_output)
        results = []
        
        if not blocks:
            # If no structured blocks, return raw code without writing to disk
            results.append({
                'note': 'No file blocks detected in LLM output. Raw code returned in response.',
                'raw_code_preview': llm_output[:500]
            })
            return results
        
        for filepath, content in blocks:
            # Sanitize the path from the LLM
            relative_path = self._sanitize_filepath(filepath, target_path)
            
            # Resolve relative to target
            full_path = self._resolve_path(target_path, relative_path)
            
            # Write the file
            write_result = self._write_file(full_path, content)
            write_ok = write_result.get('status') == 'written' or 'error' not in write_result
            
            # Verify by reading back
            verification = self._verify_file(full_path) if write_ok else {'verified': False, 'error': 'Write failed'}
            
            results.append({
                'file': relative_path,
                'original_path_from_llm': filepath,
                'resolved_path': full_path,
                'write_status': write_result.get('status', 'error'),
                'write_error': write_result.get('error', ''),
                'size': write_result.get('size', 0),
                'verified': verification.get('verified', False),
                'preview': verification.get('content_preview', '')
            })
        
        return results
    
    # ── Mode 1: Project Initialization ──────────────────────────────
    
    def _initialize_project(self, task_data: Dict) -> Dict:
        """Mode 1: Initialize a TanStack Query project structure."""
        task = task_data.get('task') or task_data.get('goal') or task_data.get('requirements', '')
        target_path = task_data.get('target_path') or task_data.get('path', '')
        framework = task_data.get('framework', 'nextjs')
        
        if not target_path:
            return {
                'status': 'needs_path',
                'message': 'Please provide a target directory path where the project should be created.',
                'mode': 'project_initialization'
            }
        
        # Check if target exists and scan existing files
        target_exists = self._check_directory_exists(target_path)
        existing_files = {}
        if target_exists:
            existing_files = self._scan_existing_structure(target_path)
        
        context_summary = self._build_context_summary(target_path, existing_files)
        
        # Retrieve memory context from past generations
        memory_context = self._retrieve_memory_context(task)
        
        # ── Step 1: Generate the project structure ──
        prompt = (
            f"Initialize a TanStack Query project structure for: {task}\n\n"
            f"Target directory: {target_path}\n"
            f"Framework: {framework}\n\n"
        )
        
        if memory_context:
            prompt += f"\n{memory_context}\n"
        
        if existing_files:
            prompt += f"The target directory already exists. Here are the existing files:\n{context_summary}\n\n"
            prompt += "Generate files that integrate with the existing project. Add or update files as needed.\n"
        else:
            prompt += "The target directory does not exist yet (or is empty). Create a fresh project structure.\n"
        
        if 'next' in framework.lower():
            prompt += (
                "\n- Use 'use client' directive for App Router components\n"
                "- Create files under app/ or src/ based on project convention\n"
                "- Provide a TanStack Query Provider wrapper for the layout\n"
            )
        elif 'react' in framework.lower():
            prompt += (
                "\n- Standard React SPA structure\n"
                "- Wrap app with QueryClientProvider in index.tsx or App.tsx\n"
            )
        
        prompt += """
Generate the following files using the `filepath:` format:

- src/connections/client.ts: Shared API client with .get<T>(), .post<T>(), etc.
- src/connections/Example/function.ts: Example pure API functions using client
- src/connections/Example/index.ts: Example TanStack Query hooks
- src/constants/queryKeys.ts: Query key factory
- src/types/index.ts: Base types and CustomError class

Use this format:
```filepath:src/connections/client.ts
// code
```
"""
        
        # Truncate prompt if too long for 8k context (~7000 chars for user, rest for system + output)
        if len(prompt) > 14000:
            prompt = prompt[:14000] + "\n\n[Prompt truncated due to length]"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        generated = call_llm(messages)
        
        # ── Step 2: Write files to disk ──
        written_files = self._write_generated_files(target_path, generated)
        
        result = {
            'status': 'initialized',
            'target_path': target_path,
            'framework': framework,
            'mode': 'project_initialization',
            'existing_files_scanned': len(existing_files),
            'files_written': written_files,
            'raw_generated_code': generated
        }
        
        # Store in memory for future learning
        self._store_in_memory('initialize', task_data, result)
        
        return result
    
    # ── Mode 2: API Ingestion & Full Stack Typing ──────────────────
    
    def _ingest_api_endpoint(self, task_data: Dict) -> Dict:
        """Mode 2: Ingest API endpoint and generate full typing stack."""
        task = task_data.get('task') or task_data.get('goal') or task_data.get('requirements', '')
        target_path = task_data.get('target_path') or task_data.get('path', '')
        api_endpoint = task_data.get('api_endpoint', '')
        payload = task_data.get('payload', '')
        response_example = task_data.get('response_example', '')
        http_method = task_data.get('http_method', 'GET')
        
        if not target_path:
            return {
                'status': 'needs_path',
                'message': 'Please provide a target directory path where the generated files should be created.',
                'mode': 'api_ingestion'
            }
        
        # Ensure shared client.ts exists
        self._ensure_shared_client(target_path)
        
        # Read existing files for context
        existing_files = self._scan_existing_structure(target_path)
        context_summary = self._build_context_summary(target_path, existing_files)
        
        # Retrieve memory context from past generations
        memory_context = self._retrieve_memory_context(task, api_endpoint)
        
        # Extract endpoint name for file naming
        # Handle both "GET https://api.example.com/endpoint" and just "https://api.example.com/endpoint"
        if ' ' in api_endpoint:
            # Split to get just the URL part
            api_endpoint = api_endpoint.split(' ', 1)[1].strip()
        
        endpoint_name = api_endpoint.strip('/').split('/')[-1] or 'api'
        if endpoint_name.startswith(':'):
            endpoint_name = f"by_{api_endpoint.strip('/').split('/')[-2]}"
        endpoint_name = endpoint_name.replace('-', '_').replace(' ', '_')
        
        prompt = (
            f"Ingest API endpoint and generate full typing stack for: {task}\n\n"
            f"Target directory: {target_path}\n"
            f"API Endpoint: {api_endpoint}\n"
            f"HTTP Method: {http_method}\n"
        )
        
        if memory_context:
            prompt += f"\n{memory_context}\n"
        
        if payload:
            prompt += f"Request Payload:\n{payload}\n"
        if response_example:
            prompt += f"Response Example (captured from real API):\n{response_example}\n"
            prompt += "\nIMPORTANT: Use the actual response structure above to generate accurate TypeScript types.\n"
        
        prompt += f"\nExisting project context:\n{context_summary}\n"
        
        # Convert endpoint_name to PascalCase for folder
        module_name = endpoint_name.replace('_', ' ').title().replace(' ', '')
        if not module_name:
            module_name = 'Api'
        
        prompt += f"""
Generate these 3 files with the `filepath:` format:

- src/types/{endpoint_name}.ts: TypeScript interfaces based on the actual API response
- src/connections/{module_name}/function.ts: Pure API functions using client
- src/connections/{module_name}/index.ts: TanStack Query hooks

IMPORTANT TYPE GENERATION RULES:
- Analyze the response_example carefully to create accurate TypeScript interfaces
- Handle nested objects, arrays, and optional fields correctly
- Include both success and error response types if available
- Use proper TypeScript generics for type safety

IMPORTANT CLIENT USAGE:
- DO NOT generate client.ts - it's a shared file at src/connections/client.ts
- Import client from "../client" in function.ts: import {{ client }} from "../client"
- All modules share the same client for common functionality (auth, error handling, etc.)

```filepath:src/types/{endpoint_name}.ts
// TypeScript interfaces based on actual API response
```
```filepath:src/connections/{module_name}/function.ts
// Pure API functions using client - NO @tanstack/react-query imports
// Import: import {{ client }} from "../client"
```
```filepath:src/connections/{module_name}/index.ts
// TanStack Query hooks with proper error handling and toast notifications
```
"""
        
        # Truncate prompt if too long for 8k context
        if len(prompt) > 14000:
            prompt = prompt[:14000] + "\n\n[Prompt truncated due to length]"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        generated = call_llm(messages)
        
        # Write files to disk
        written_files = self._write_generated_files(target_path, generated)
        
        result = {
            'status': 'ingested',
            'target_path': target_path,
            'api_endpoint': api_endpoint,
            'http_method': http_method,
            'mode': 'api_ingestion',
            'existing_files_scanned': len(existing_files),
            'files_written': written_files,
            'raw_generated_code': generated,
            'module_name': module_name,
            'endpoint_name': endpoint_name
        }
        
        # Store in memory for future learning
        self._store_in_memory('ingest_api', task_data, result)
        
        return result
    
    # ── Hook / Types / Service generation ──────────────────────────
    
    def _generate_hook(self, task_data: Dict) -> Dict:
        """Generate an index.ts file (TanStack Query hooks) following the Connection Structure Guide."""
        task = task_data.get('task') or task_data.get('goal') or task_data.get('requirements', '')
        target_path = task_data.get('target_path') or task_data.get('path', '')
        hook_type = task_data.get('hook_type', 'query')  # 'query' or 'mutation'
        functions_file = task_data.get('functions_file', '')
        types_file = task_data.get('types_file', '')
        query_key_name = task_data.get('query_key_name', '')
        module_name = task_data.get('module_name', '')
        
        if not target_path:
            return {
                'status': 'needs_path',
                'message': 'Please provide a target directory path where the index.ts should be created.'
            }
        
        existing_files = self._scan_existing_structure(target_path)
        context_summary = self._build_context_summary(target_path, existing_files)
        
        # Derive module name from task
        if not module_name:
            words = task.split()
            raw = words[0].lower() if words else 'resource'
            if raw in ('create', 'make', 'build', 'add', 'new', 'get', 'fetch', 'set', 'use', 'generate'):
                raw = words[1].lower() if len(words) > 1 else 'resource'
            module_name = raw.rstrip('s').title()  # PascalCase, singular
        
        module_key = module_name.lower()  # For queryKeys.moduleName
        
        prompt = (
            f"Generate an index.ts file (TanStack Query hooks) for: {task}\n\n"
            f"Target: src/connections/{module_name}/index.ts\n"
            f"Hook type: {hook_type}\n\n"
            f"IMPORTANT: This file must follow the Connection Structure Guide pattern:\n"
            f"- Import {{ useQuery, useMutation, useQueryClient, type UseQueryOptions }} from '@tanstack/react-query'\n"
            f"- Import {{ premiumToast }} from \"../../components/ui/feedback\"\n"
            f"- Import functions from ./function (e.g., getRequest, createRequest)\n"
            f"- Import {{ queryKeys }} from \"../../constants/queryKeys\"\n"
            f"- Import type {{ CustomError }} from \"../../utils/error\"\n"
            f"- Import types from ../../types/{module_key}\n"
            f"- Hook naming: use{module_name}Query (queries), useCreate{module_name}Mutation (creates), useUpdate{module_name}Mutation (updates), useDelete{module_name}Mutation (deletes)\n"
            f"- For queries: use queryKey: queryKeys.{module_key}.list(filters) and queryFn\n"
            f"- For mutations: use mutationFn, onSuccess (invalidateQueries + premiumToast.success), onError (premiumToast.error)\n"
            f"- Use queryClient.invalidateQueries({{ queryKey: queryKeys.{module_key}.lists() }}) for list invalidation\n"
            f"- Use queryClient.invalidateQueries({{ queryKey: queryKeys.{module_key}.detail(id) }}) for detail invalidation\n"
            f"- For error handling: check error.statusCode (e.g., skip toast for 403)\n"
            f"- For store integration: import from ../../store/authStore etc.\n"
            f"- Add JSDoc comments on every exported hook\n"
            f"- Use TypeScript generics for type safety throughout\n"
        )
        
        if functions_file:
            prompt += f"\nFunctions file content:\n{functions_file}\n"
        if types_file:
            prompt += f"\nTypes file content:\n{types_file}\n"
        if query_key_name:
            prompt += f"\nQuery key name to use: {query_key_name}\n"
        
        prompt += f"\nExisting project context:\n{context_summary}\n"
        prompt += f"\nWrite the index.ts file using this format:\n\n```filepath:src/connections/{module_name}/index.ts\n// TanStack Query hooks - imports from @tanstack/react-query and ./function\n```\n"
        
        # Truncate prompt if too long for 8k context
        if len(prompt) > 14000:
            prompt = prompt[:14000] + "\n\n[Prompt truncated due to length]"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        generated = call_llm(messages)
        written_files = self._write_generated_files(target_path, generated)
        
        return {
            'status': 'generated',
            'target_path': target_path,
            'module_name': module_name,
            'file_type': 'index.ts',
            'hook_type': hook_type,
            'files_written': written_files,
            'raw_generated_code': generated
        }
    
    def _generate_types(self, task_data: Dict) -> Dict:
        """Generate TypeScript type definitions and write them to disk."""
        task = task_data.get('task') or task_data.get('goal') or task_data.get('requirements', '')
        target_path = task_data.get('target_path') or task_data.get('path', '')
        payload = task_data.get('payload', '')
        response_example = task_data.get('response_example', '')
        
        if not target_path:
            return {
                'status': 'needs_path',
                'message': 'Please provide a target directory path where the types should be created.'
            }
        
        existing_files = self._scan_existing_structure(target_path)
        context_summary = self._build_context_summary(target_path, existing_files)
        
        # Generate a type name from the task
        type_name = ''.join(w.capitalize() for w in task.split()[:3] if w.isalnum())
        
        prompt = (
            f"Generate TypeScript type definitions for: {task}\n\n"
            f"Target directory: {target_path}/src/types/\n"
        )
        
        if payload:
            prompt += f"\nRequest Payload/Schema:\n{payload}\n"
        if response_example:
            prompt += f"\nResponse Example:\n{response_example}\n"
        
        prompt += f"\nExisting project context:\n{context_summary}\n"
        prompt += f"\nWrite the types file using this format (ONLY interfaces, NO @tanstack/react-query):\n\n```filepath:src/types/{type_name}.ts\n// TypeScript types only\n```\n"
        
        # Truncate prompt if too long for 8k context
        if len(prompt) > 14000:
            prompt = prompt[:14000] + "\n\n[Prompt truncated due to length]"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        generated = call_llm(messages)
        written_files = self._write_generated_files(target_path, generated)
        
        return {
            'status': 'generated',
            'target_path': target_path,
            'type_name': type_name,
            'files_written': written_files,
            'raw_generated_code': generated
        }
    
    def _generate_functions(self, task_data: Dict) -> Dict:
        """Generate a function.ts file following the Connection Structure Guide patterns."""
        task = task_data.get('task') or task_data.get('goal') or task_data.get('requirements', '')
        target_path = task_data.get('target_path') or task_data.get('path', '')
        api_endpoint = task_data.get('api_endpoint', '')
        http_method = task_data.get('http_method', 'GET')
        types_import = task_data.get('types_import', '')
        
        if not target_path:
            return {
                'status': 'needs_path',
                'message': 'Please provide a target directory path where the function.ts should be created.'
            }
        
        existing_files = self._scan_existing_structure(target_path)
        context_summary = self._build_context_summary(target_path, existing_files)
        
        module_name = api_endpoint.strip('/').split('/')[-1] or 'Api'
        module_name = module_name.replace('-', ' ').replace('_', ' ').title().replace(' ', '')
        if module_name.startswith(':'):
            module_name = 'By' + api_endpoint.strip('/').split('/')[-2].title()
        
        prompt = (
            f"Generate a function.ts file for: {task}\n\n"
            f"Target: src/connections/{module_name}/function.ts\n"
            f"API Endpoint: {api_endpoint}\n"
            f"HTTP Method: {http_method}\n\n"
            f"IMPORTANT: This file must follow the Connection Structure Guide pattern:\n"
            f"- Import {{ client }} from \"../client\" (SHARED client, do not create client.ts)\n"
            f"- Import types from ../../types/ or ../../types/{endpoint_name.lower()}\n"
            f"- Use client.get<T>(), client.post<T>(), client.put<T>(), client.patch<T>(), or client.delete<T>()\n"
            f"- Include {{ includeAuth: true }} option on all requests\n"
            f"- Use URLSearchParams for building query params on GET requests\n"
            f"- Add JSDoc comments on every exported function\n"
            f"- Use isFormData: true for file uploads\n"
            f"- Use errorMessage option for user-friendly error messages\n"
            f"- Use responseType: \"blob\" for file downloads\n"
            f"- Do NOT import from @tanstack/react-query in this file\n"
            f"- DO NOT generate client.ts - it's a shared file at src/connections/client.ts\n"
        )
        
        if types_import:
            prompt += f"\nTypes to use:\n{types_import}\n"
        
        prompt += f"\nExisting project context:\n{context_summary}\n"
        prompt += f"\nWrite the function.ts file using this format:\n\n```filepath:src/connections/{module_name}/function.ts\n// Pure API functions using client - NO @tanstack/react-query imports\n```\n"
        
        # Truncate prompt if too long for 8k context
        if len(prompt) > 14000:
            prompt = prompt[:14000] + "\n\n[Prompt truncated due to length]"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        generated = call_llm(messages)
        written_files = self._write_generated_files(target_path, generated)
        
        return {
            'status': 'generated',
            'target_path': target_path,
            'module_name': module_name,
            'file_type': 'function.ts',
            'api_endpoint': api_endpoint,
            'http_method': http_method,
            'files_written': written_files,
            'raw_generated_code': generated
        }
    
    # ── Refactor / Debug / Explain ─────────────────────────────────
    
    def _refactor_tanstack(self, task_data: Dict) -> Dict:
        """Refactor existing TanStack Query code."""
        code = task_data.get('code', '')
        file_path = task_data.get('file_path', '')
        requirements = task_data.get('requirements', 'improve type safety and follow TanStack Query v5+ best practices')
        
        # Read file if path provided but no code
        if file_path and not code:
            content = self._read_file_content(file_path)
            if content:
                code = content
        
        user_msg = f"Refactor this TanStack Query code to {requirements}:\n\n{code}\n\nProvide the refactored code and explain the improvements made."
        if len(user_msg) > 14000:
            user_msg = user_msg[:14000] + "\n\n[Prompt truncated due to length]"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg}
        ]
        
        refactored = call_llm(messages)
        
        result = {
            'status': 'refactored',
            'refactored_code': refactored
        }
        
        # If a file path was given, write the refactored code back
        if file_path:
            blocks = self._extract_file_blocks(refactored)
            if blocks:
                # Use the first block's content
                _, new_content = blocks[0]
                write_result = self._write_file(file_path, new_content)
                verification = self._verify_file(file_path)
                result['written'] = write_result
                result['verified'] = verification
            else:
                # Write the whole output back
                write_result = self._write_file(file_path, refactored)
                result['written'] = write_result
        
        return result
    
    def _debug_tanstack(self, task_data: Dict) -> Dict:
        """Debug TanStack Query code."""
        code = task_data.get('code', '')
        file_path = task_data.get('file_path', '')
        error = task_data.get('error', '')
        
        # Read file if path provided but no code
        if file_path and not code:
            content = self._read_file_content(file_path)
            if content:
                code = content
        
        user_msg = f"Debug this TanStack Query code:\n\n{code}\n\nError: {error}\n\nIdentify the issue and provide the fix."
        if len(user_msg) > 14000:
            user_msg = user_msg[:14000] + "\n\n[Prompt truncated due to length]"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg}
        ]
        
        debug_result = call_llm(messages)
        
        return {
            'status': 'debugged',
            'fix': debug_result,
            'file_path': file_path
        }
    
    def _explain_tanstack(self, task_data: Dict) -> Dict:
        """Explain TanStack Query code or concepts."""
        code = task_data.get('code', '')
        file_path = task_data.get('file_path', '')
        concept = task_data.get('concept', '')
        
        # Read file if path provided but no code
        if file_path and not code:
            content = self._read_file_content(file_path)
            if content:
                code = content
        
        prompt_parts = []
        if code:
            prompt_parts.append(f"Explain this TanStack Query code:\n\n{code}\n")
        if concept:
            prompt_parts.append(f"Explain this TanStack Query concept: {concept}\n")
        prompt_parts.append("Provide a clear explanation with examples where helpful.")
        
        prompt = '\n'.join(prompt_parts)
        
        if len(prompt) > 14000:
            prompt = prompt[:14000] + "\n\n[Prompt truncated due to length]"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        explanation = call_llm(messages)
        
        return {
            'status': 'explained',
            'explanation': explanation,
            'file_path': file_path
        }
    
    def get_status(self) -> str:
        """Get current agent status."""
        return self.status
