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
        self._memory_cache = {}  # Simple cache for memory search results
        self._auto_analyze = False  # Opt-in for automatic analysis (default False for performance)
        self.system_prompt = """You are a TanStack Query (v5+) & TypeScript Generation Agent. Generate code following this architecture:

ARCHITECTURE: src/connections/{Module}/function.ts + index.ts (PascalCase folders, singular names)
- function.ts: Pure API calls via client.get<T>()/post<T>() etc. NO @tanstack/react-query imports
- index.ts: TanStack Query hooks using ./function. Import { useQuery, useMutation, useQueryClient, type UseQueryOptions } from '@tanstack/react-query'
- client.ts: SHARED client at src/connections/client.ts (common for all modules) with .get<T>(), .post<T>(), .put<T>(), .patch<T>(), .delete<T>()
- Types: In src/types/ folder (centralized)
- Query keys: Centralized in src/constants/queryKeys.ts with pattern keys.module.list(filters), .lists(), .detail(id)
- Error: CustomError class with statusCode
- Toast: premiumToast from ../../components/ui/feedback (ALWAYS use this for all success/error messages)
- Store: Zustand stores from ../../store/
- QueryClientProvider: In main app file (not separate provider)

IMPORTANT RULES:
1. client.ts is a SHARED file at src/connections/client.ts. All modules import from this common client.
   Import path: import { client } from "../client" (from function.ts in src/connections/{Module}/)

2. Hook naming: useXQuery (queries), useXMutation (mutations)

3. Toast usage (MANDATORY for all mutations):
   - Import: import { premiumToast } from "../../components/ui/feedback"
   - Success: premiumToast.success("Success message")
   - Error: premiumToast.error(error.message || "Fallback error message")
   - Loading: premiumToast.loading("Loading message", progress)
   - NEVER use console.log, console.error, or other toast libraries for user-facing messages

4. Mutation pattern:
   - onSuccess: invalidates queries + premiumToast.success("Success message")
   - onError: premiumToast.error(error.message || "Error message")

5. Query key pattern in src/constants/queryKeys.ts:
   export const queryKeys = {
     moduleName: {
       list: (filters?: unknown) => ["moduleName", "list", filters] as const,
       lists: () => ["moduleName", "list"] as const,
       detail: (id: string) => ["module", id] as const,
     },
   }

6. function.ts structure:
   - Import types from ../../types (NOT inline)
   - Import client from "../client"
   - Helper functions for query params (buildQueryParams)
   - JSDoc comments for each function
   - NO @tanstack/react-query imports

7. index.ts structure:
   - Import { useQuery, useMutation, useQueryClient, type UseQueryOptions } from '@tanstack/react-query'
   - Import { premiumToast } from "../../components/ui/feedback"
   - Import functions from ./function
   - Import queryKeys from "../../constants/queryKeys"
   - Import CustomError from "../../utils/error"
   - Import types from "../../types"

8. Type handling:
   - Always import types from centralized type files
   - Use ApiResponse<T> wrapper for responses
   - Use CustomError for error typing
   - DO NOT define types inline in function.ts

9. Client options:
   - includeAuth: boolean
   - isFormData: boolean (for file uploads)
   - customHeaders: Record<string, string>
   - errorMessage: string (for toast)
   - params: Record<string, unknown>
   - responseType: "json" | "blob" | "text"
   - authType: "staff" | "storefront"

10. Query invalidation:
    - Use queryKeys.module.lists() for invalidating all list variants
    - Use queryKeys.module.detail(id) for specific resource
    - Call refreshNotifications(queryClient) after mutations where applicable

11. File patterns:
    - GET: build URLSearchParams for filters, append to URL
    - POST/PUT/PATCH: include errorMessage option
    - DELETE: include errorMessage option
    - FormData: set isFormData: true
    - Blob download: set responseType: "blob"

OUTPUT FORMAT: Use ```filepath:path/to/file\ncode``` blocks. Each block = one file.""" 
    
    def execute(self, task_data: Dict) -> Dict:
        """Execute a TanStack Query task."""
        self.status = "working"
        task_type = task_data.get('type')
        
        # Handle special task types for self-improvement
        if task_type == 'analyze_improvements':
            target_path = task_data.get('target_path') or task_data.get('path', '')
            result = self.analyze_and_suggest_improvements(target_path)
            self.status = "idle"
            return result
        
        if task_type == 'apply_suggestion':
            suggestion_id = task_data.get('suggestion_id', '')
            approval = task_data.get('approval', False)
            result = self.apply_suggestion(suggestion_id, approval)
            self.status = "idle"
            return result
        
        # Handle test operations
        if task_type == 'check_packages':
            target_path = task_data.get('target_path') or task_data.get('path', '')
            result = self._check_required_packages(target_path)
            self.status = "idle"
            return {'status': 'checked', 'package_check': result}
        
        if task_type == 'set_auto_analyze':
            enabled = task_data.get('enabled', False)
            self._auto_analyze = enabled
            self.status = "idle"
            return {'status': 'success', 'success': True, 'auto_analyze': enabled}
        
        if task_type == 'test_cache':
            query = task_data.get('query', '')
            endpoint = task_data.get('endpoint', '')
            # Test memory retrieval with caching
            result = self._retrieve_memory_context(query, endpoint)
            self.status = "idle"
            return {'status': 'tested', 'cached': result != '', 'result': result}
        
        if task_type == 'clear_cache':
            self._memory_cache = {}
            self.status = "idle"
            return {'status': 'success', 'success': True}
        
        if task_type == 'get_boilerplate':
            template_type = task_data.get('template_type', 'customError')
            
            template = ''
            if template_type == 'customError':
                template = self._get_boilerplate_custom_error()
            elif template_type == 'client':
                template = self._get_boilerplate_client()
            elif template_type == 'apiClient':
                template = self._get_boilerplate_api_client()
            elif template_type == 'premiumToast':
                template = self._get_boilerplate_premium_toast()
            elif template_type == 'premiumToastHelper':
                template = self._get_boilerplate_premium_toast_helper()
            
            self.status = "idle"
            return {'status': 'generated', 'template': template, 'type': template_type}
        
        if task_type == 'test_truncation':
            instructions = task_data.get('instructions', '')
            context_sections = task_data.get('context_sections', [])
            max_length = task_data.get('max_length', 14000)
            
            result = self._build_prompt_with_truncation(instructions, context_sections, max_length)
            
            self.status = "idle"
            return {
                'status': 'tested',
                'total_length': len(result),
                'instructions_length': len(instructions),
                'context_length': len(result) - len(instructions) - 2,  # Subtract newline
                'instructions_preserved': instructions in result,
                'result': result[:500] + '...' if len(result) > 500 else result
            }
        
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
            
            # Trigger automatic analysis after successful generations (only if opt-in)
            if self._auto_analyze and result.get('status') in ['initialized', 'ingested', 'generated'] and self.memory_agent:
                try:
                    target_path = task_data.get('target_path') or task_data.get('path', '')
                    # Store analysis result in result for user to review
                    analysis = self.analyze_and_suggest_improvements(target_path)
                    if analysis.get('status') == 'analyzed' and analysis.get('total_suggestions', 0) > 0:
                        result['improvement_suggestions'] = analysis
                except Exception:
                    # Don't fail the main operation if analysis fails
                    pass
            
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
import { API_BASE_URL } from '../config/api.config';

/**
 * Shared HTTP client for all API connections.
 * Provides typed methods for common HTTP operations with built-in error handling.
 * 
 * Base URL Configuration:
 * - Set NEXT_PUBLIC_API_URL in your .env file (for Next.js/Vite)
 * - Set VITE_API_URL in your .env file (for Vite)
 * - Falls back to http://localhost:8000 if not set
 * 
 * Example .env file:
 * NEXT_PUBLIC_API_URL=https://api.example.com
 * or
 * VITE_API_URL=https://api.example.com
 */
class ApiClient {
  private client: AxiosInstance;
  private baseURL: string;

  constructor() {
    // Use the centralized config for base URL
    this.baseURL = API_BASE_URL;
    
    this.client = axios.create({
      baseURL: this.baseURL,
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
  
  /**
   * Update the base URL dynamically (useful for multi-tenant apps)
   */
  setBaseURL(newBaseURL: string): void {
    this.baseURL = newBaseURL;
    this.client.defaults.baseURL = newBaseURL;
  }
  
  /**
   * Get the current base URL
   */
  getBaseURL(): string {
    return this.baseURL;
  }
}

// Export singleton instance
export const client = new ApiClient();
"""
        
        # Write the client.ts file
        self._write_file(client_path, client_content)
        
        # Also create the config file
        self._ensure_api_config(target_path)
    
    def _ensure_api_config(self, target_path: str):
        """Ensure the API config file exists at src/config/api.config.ts."""
        config_path = f"{target_path}/src/config/api.config.ts"
        
        # Check if config already exists
        existing_content = self._read_file_content(config_path)
        if existing_content:
            return  # Already exists
        
        # Create the config file
        config_content = """/**
 * API Configuration
 * Centralized configuration for API base URL and other API-related settings.
 */

// Support multiple env variable naming conventions
export const API_BASE_URL = 
  (typeof process !== 'undefined' && process.env?.NEXT_PUBLIC_API_URL) ||
  (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL) ||
  'http://localhost:8000';

// API timeout in milliseconds
export const API_TIMEOUT = 30000;

// API version (if applicable)
export const API_VERSION = 'v1';

// Default headers
export const DEFAULT_HEADERS = {
  'Content-Type': 'application/json',
};
"""
        
        # Write the config file
        self._write_file(config_path, config_content)

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
        """Retrieve relevant past memory context for a task with caching."""
        if not self.memory_agent:
            return ''
        
        # Create cache key
        cache_key = f"{task}_{endpoint}"
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        
        try:
            memories = []
            
            # Combined single search for better performance
            query = f"{task} {endpoint}".strip()
            code_results = self.memory_agent.search_memory(
                query=query,
                collection='code',
                n_results=3
            )
            
            if code_results.get('status') == 'searched' and code_results.get('results'):
                for i, doc in enumerate(code_results['results']):
                    meta = code_results['metadatas'][i] if i < len(code_results['metadatas']) else {}
                    score = code_results['distances'][i] if i < len(code_results['distances']) else 0
                    if score < 1.5:  # Only include relevant memories
                        # Check if it's an API pattern
                        if meta.get('endpoint'):
                            memories.append(f"- Similar API pattern: {meta.get('method', '')} {meta.get('endpoint', '')} (module: {meta.get('module_name', '')})")
                        else:
                            memories.append(f"- Previous generation: {doc}")
            
            result = ''
            if memories:
                result = 'From past work:\n' + '\n'.join(memories) + '\n'
            
            # Cache the result
            self._memory_cache[cache_key] = result
            return result
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
    
    def analyze_and_suggest_improvements(self, target_path: str = '') -> Dict:
        """Analyze memory and suggest improvements while respecting core base immutability.
        
        This method:
        1. Analyzes past generations and corrections from memory
        2. Identifies patterns and potential improvements
        3. Generates suggestions that DO NOT change the core Connection Structure Guide
        4. Returns suggestions for user approval
        
        Core base (immutable):
        - src/connections/{Module}/function.ts + index.ts structure
        - Query key patterns
        - Import patterns
        - Client usage patterns
        - Error handling patterns
        
        Suggested improvements (mutable):
        - Specific file content optimizations
        - Helper function additions
        - Type definition refinements
        - Hook pattern variations
        - Query invalidation strategies
        
        Args:
            target_path: Optional target directory to analyze
        
        Returns:
            Dict with suggestions list and metadata
        """
        if not self.memory_agent:
            return {'error': 'Memory agent not available', 'status': 'error'}
        
        suggestions = []
        
        try:
            # Analyze corrections collection for patterns
            corrections = self.memory_agent.search_memory(
                query='correction improvement pattern',
                collection='corrections',
                n_results=10
            )
            
            if corrections.get('status') == 'searched' and corrections.get('results'):
                # Group corrections by context to find patterns
                correction_patterns = {}
                for i, doc in enumerate(corrections['results']):
                    meta = corrections['metadatas'][i] if i < len(corrections['metadatas']) else {}
                    context = meta.get('context', 'general')
                    if context not in correction_patterns:
                        correction_patterns[context] = []
                    correction_patterns[context].append(doc)
                
                # Generate suggestions from patterns
                for context, docs in correction_patterns.items():
                    if len(docs) >= 2:  # Only suggest if pattern appears multiple times
                        suggestions.append({
                            'type': 'pattern_improvement',
                            'context': context,
                            'occurrences': len(docs),
                            'suggestion': f"Multiple corrections detected for '{context}'. Consider standardizing this pattern.",
                            'priority': 'medium',
                            'affects_core': False
                        })
            
            # Analyze API patterns for common structures
            api_patterns = self.memory_agent.search_memory(
                query='API endpoint module structure',
                collection='api_patterns',
                n_results=20
            )
            
            if api_patterns.get('status') == 'searched' and api_patterns.get('results'):
                # Check for inconsistent naming patterns
                module_names = []
                for i, meta in enumerate(api_patterns.get('metadatas', [])):
                    module_name = meta.get('module_name', '')
                    if module_name:
                        module_names.append(module_name)
                
                # Check for naming inconsistencies
                if module_names:
                    singular_count = sum(1 for name in module_names if not name.endswith('s'))
                    plural_count = sum(1 for name in module_names if name.endswith('s'))
                    
                    if singular_count > 0 and plural_count > 0:
                        suggestions.append({
                            'type': 'naming_consistency',
                            'suggestion': f"Inconsistent module naming detected: {singular_count} singular, {plural_count} plural. Consider standardizing to singular names as per Connection Structure Guide.",
                            'priority': 'low',
                            'affects_core': False,
                            'current_state': {'singular': singular_count, 'plural': plural_count}
                        })
            
            # Analyze code collection for common patterns
            code_patterns = self.memory_agent.search_memory(
                query='function index hook query mutation',
                collection='code',
                n_results=15
            )
            
            if code_patterns.get('status') == 'searched' and code_patterns.get('results'):
                # Check for missing query keys
                missing_keys_count = 0
                for i, doc in enumerate(code_patterns['results']):
                    if 'queryKeys' not in doc and 'query_keys' not in doc.lower():
                        missing_keys_count += 1
                
                if missing_keys_count > 0:
                    suggestions.append({
                        'type': 'query_key_usage',
                        'suggestion': f"{missing_keys_count} generations may not be using centralized queryKeys. Ensure all hooks use queryKeys from src/constants/queryKeys.ts",
                        'priority': 'high',
                        'affects_core': False
                    })
            
            # Generate suggestions for target-specific improvements
            if target_path:
                existing_files = self._scan_existing_structure(target_path)
                
                # Check for missing queryKeys file
                if 'src/constants/queryKeys.ts' not in existing_files:
                    suggestions.append({
                        'type': 'missing_file',
                        'suggestion': "queryKeys.ts not found in src/constants/. Consider creating centralized query keys following the Connection Structure Guide pattern.",
                        'priority': 'high',
                        'affects_core': False,
                        'file': 'src/constants/queryKeys.ts'
                    })
                
                # Check for missing CustomError
                has_custom_error = False
                for filepath, content in existing_files.items():
                    if 'CustomError' in content:
                        has_custom_error = True
                        break
                
                if not has_custom_error:
                    suggestions.append({
                        'type': 'missing_type',
                        'suggestion': "CustomError class not found in existing files. Consider adding it to src/utils/error.ts following the Connection Structure Guide.",
                        'priority': 'medium',
                        'affects_core': False
                    })
            
            return {
                'status': 'analyzed',
                'suggestions': suggestions,
                'total_suggestions': len(suggestions),
                'metadata': {
                    'corrections_analyzed': len(corrections.get('results', [])) if corrections.get('results') else 0,
                    'api_patterns_analyzed': len(api_patterns.get('results', [])) if api_patterns.get('results') else 0,
                    'code_patterns_analyzed': len(code_patterns.get('results', [])) if code_patterns.get('results') else 0,
                    'core_base_unchanged': True,
                    'timestamp': str(__import__('datetime').datetime.now())
                }
            }
            
        except Exception as e:
            return {'error': str(e), 'status': 'error'}
    
    def apply_suggestion(self, suggestion_id: str, approval: bool) -> Dict:
        """Apply or decline a suggested improvement.
        
        This method:
        1. Takes a suggestion ID and user approval decision
        2. If approved: applies the change (only to mutable parts, never core base)
        3. If declined: records the decline for future learning
        4. Stores the decision in memory
        
        Args:
            suggestion_id: ID of the suggestion to process
            approval: True to apply, False to decline
        
        Returns:
            Dict with status and details of the action taken
        """
        if not self.memory_agent:
            return {'error': 'Memory agent not available', 'status': 'error'}
        
        try:
            # Store the decision in memory
            decision_content = f"Suggestion {suggestion_id}: {'APPROVED' if approval else 'DECLINED'}"
            
            self.memory_agent.save_memory(
                content=decision_content,
                collection='suggestion_decisions',
                metadata={
                    'suggestion_id': suggestion_id,
                    'approved': approval,
                    'timestamp': str(__import__('datetime').datetime.now())
                }
            )
            
            if approval:
                # Apply the suggestion (implementation depends on suggestion type)
                # This is a placeholder - actual implementation would be suggestion-specific
                return {
                    'status': 'approved',
                    'suggestion_id': suggestion_id,
                    'message': 'Suggestion approved. Implementation would be suggestion-specific.',
                    'note': 'Core base remains unchanged as per immutability rule.'
                }
            else:
                return {
                    'status': 'declined',
                    'suggestion_id': suggestion_id,
                    'message': 'Suggestion declined. Decision recorded for learning.',
                    'note': 'Decline pattern will be considered in future suggestions.'
                }
                
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
    
    def _build_prompt_with_truncation(self, instructions: str, context_sections: List[str], max_length: int = 14000) -> str:
        """Build a prompt with smart truncation that preserves instructions.
        
        Args:
            instructions: The core instructions/format block that must be preserved
            context_sections: List of context sections (memory, existing files, etc.)
            max_length: Maximum total prompt length
        
        Returns:
            Truncated prompt with instructions preserved
        """
        # Reserve space for instructions
        instructions_length = len(instructions)
        context_budget = max_length - instructions_length - 100  # 100 char buffer
        
        if context_budget <= 0:
            # Even instructions are too long, truncate them as fallback
            return instructions[:max_length] + "\n\n[Instructions truncated due to length]"
        
        # Build context from sections, truncating if needed
        context_parts = []
        current_length = 0
        
        for section in context_sections:
            if current_length + len(section) <= context_budget:
                context_parts.append(section)
                current_length += len(section)
            else:
                # Truncate this section to fit remaining budget
                remaining = context_budget - current_length
                if remaining > 50:  # Only add if we have meaningful space
                    context_parts.append(section[:remaining] + "\n...[context truncated]")
                break
        
        # Combine context and instructions
        full_prompt = '\n'.join(context_parts) + '\n\n' + instructions
        
        return full_prompt
    
    def _get_boilerplate_custom_error(self) -> str:
        """Get boilerplate CustomError class."""
        return """export class CustomError extends Error {
  statusCode: number;
  errors?: Record<string, string[]>;

  constructor(message: string, statusCode: number, errors?: Record<string, string[]>) {
    super(message);
    this.name = "CustomError";
    this.statusCode = statusCode;
    this.errors = errors;
    Object.setPrototypeOf(this, CustomError.prototype);
  }
}"""
    
    def _get_boilerplate_client(self) -> str:
        """Get boilerplate client.ts content."""
        return """/**
 * Lightweight, typed API client wrapper.
 *
 * Provides simple helpers around the project's API request utility.
 * Includes standard HTTP methods (get, post, put, patch, delete, head)
 */

import { apiRequest, API_BASE_URL, getAuthHeader } from "../utils/apiClient";
import type { ApiRequestConfig } from "../utils/apiClient";

/**
 * Configuration options for all client requests.
 */
export type RequestOptions = {
  includeAuth?: boolean;
  isFormData?: boolean;
  customHeaders?: Record<string, string>;
  redirect?: RequestRedirect;
  errorMessage?: string;
  params?: Record<string, unknown>;
  responseType?: "json" | "blob" | "text";
  authType?: "staff" | "storefront";
};

/**
 * Helper to build consistent options for API requests.
 */
const buildOptions = (
  method: "GET" | "POST" | "PUT" | "PATCH" | "DELETE" | "HEAD",
  data?: unknown,
  opts?: RequestOptions,
): ApiRequestConfig => ({
  method,
  data,
  includeAuth: opts?.includeAuth,
  isFormData: opts?.isFormData,
  customHeaders: opts?.customHeaders,
  redirect: opts?.redirect,
  errorMessage: opts?.errorMessage,
  params: opts?.params,
  responseType: opts?.responseType,
  authType: opts?.authType,
});

/**
 * Main API client instance.
 */
export const client = {
  get: async <T>(url: string, opts?: RequestOptions) =>
    apiRequest<T>(url, buildOptions("GET", undefined, opts)),

  post: async <TResponse, TData = unknown>(
    url: string,
    data?: TData,
    opts?: RequestOptions,
  ) => 
    apiRequest<TResponse>(url, buildOptions("POST", data, opts)),

  put: async <TResponse, TData = unknown>(
    url: string,
    data?: TData,
    opts?: RequestOptions,
  ) => 
    apiRequest<TResponse>(url, buildOptions("PUT", data, opts)),

  patch: async <TResponse, TData = unknown>(
    url: string,
    data?: TData,
    opts?: RequestOptions,
  ) => 
    apiRequest<TResponse>(url, buildOptions("PATCH", data, opts)),

  delete: async <TResponse, TData = unknown>(
    url: string,
    data?: TData,
    opts?: RequestOptions,
  ) => 
    apiRequest<TResponse>(url, buildOptions("DELETE", data, opts)),

  head: async <T>(url: string, opts?: RequestOptions) =>
    apiRequest<T>(url, buildOptions("HEAD", undefined, opts)),
};

export default client;"""
    
    def _get_boilerplate_api_client(self) -> str:
        """Get boilerplate apiClient.ts content."""
        return """/**
 * Core API request utility.
 * Handles fetch requests, error handling, and authentication.
 */

export interface ApiRequestConfig {
  method: "GET" | "POST" | "PUT" | "PATCH" | "DELETE" | "HEAD";
  data?: unknown;
  includeAuth?: boolean;
  isFormData?: boolean;
  customHeaders?: Record<string, string>;
  redirect?: RequestRedirect;
  errorMessage?: string;
  params?: Record<string, unknown>;
  responseType?: "json" | "blob" | "text";
  authType?: "staff" | "storefront";
}

/**
 * Base API URL - configure for your environment.
 */
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Gets the current authorization header.
 * Implement based on your auth storage mechanism.
 */
export const getAuthHeader = (): string => {
  const token = typeof window !== "undefined" 
    ? localStorage.getItem("authToken") 
    : null;
  return token ? `Bearer ${token}` : "";
};

/**
 * Core API request function.
 */
export async function apiRequest<T>(
  url: string,
  config: ApiRequestConfig,
): Promise<T> {
  const {
    method,
    data,
    includeAuth = false,
    isFormData = false,
    customHeaders = {},
    redirect,
    errorMessage,
    params,
    responseType = "json",
  } = config;

  // Build URL with query params
  let finalUrl = url.startsWith("http") ? url : `${API_BASE_URL}${url}`;
  if (params) {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        searchParams.append(key, String(value));
      }
    });
    const queryString = searchParams.toString();
    if (queryString) {
      finalUrl += `?${queryString}`;
    }
  }

  // Build headers
  const headers: Record<string, string> = {
    ...(isFormData ? {} : { "Content-Type": "application/json" }),
    ...customHeaders,
  };

  if (includeAuth) {
    headers["Authorization"] = getAuthHeader();
  }

  // Build request options
  const options: RequestInit = {
    method,
    headers,
    redirect,
    body: data 
      ? isFormData 
        ? data as FormData 
        : JSON.stringify(data)
      : undefined,
  };

  try {
    const response = await fetch(finalUrl, options);

    // Handle response based on type
    if (responseType === "blob") {
      if (!response.ok) {
        throw new Error(errorMessage || "Request failed");
      }
      return (await response.blob()) as T;
    }

    if (responseType === "text") {
      if (!response.ok) {
        throw new Error(errorMessage || "Request failed");
      }
      return (await response.text()) as T;
    }

    // JSON response
    const jsonData = await response.json();

    if (!response.ok) {
      throw new Error(
        jsonData?.message || errorMessage || "Request failed"
      );
    }

    return jsonData as T;
  } catch (error) {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error(errorMessage || "An unexpected error occurred");
  }
}"""
    
    def _get_boilerplate_premium_toast(self) -> str:
        """Get boilerplate PremiumToast.tsx content."""
        return """import React from 'react';
import { toast } from 'sonner';
import { 
  FaCheckCircle, 
  FaExclamationCircle, 
  FaInfoCircle, 
  FaTimes,
  FaCloudUploadAlt,
  FaCog,
  FaSpinner
} from 'react-icons/fa';

export type ToastType = 'success' | 'error' | 'info' | 'warning' | 'upload' | 'update' | 'message' | 'loading';

interface PremiumToastProps {
  onClose: () => void;
  type: ToastType;
  title: string;
  description?: string;
  actionLabel?: string;
  onAction?: () => void;
  avatar?: string;
  progress?: number;
  version?: string;
}

const PremiumToast: React.FC<PremiumToastProps> = ({
  onClose,
  type,
  title,
  description,
  actionLabel,
  onAction,
  avatar,
  progress,
  version,
}) => {
  const dismiss = React.useCallback(() => {
    onClose();
    setTimeout(() => {
      toast.dismiss();
    }, 0);
  }, [onClose]);

  const handleAction = () => {
    if (onAction) {
      onAction();
    }
    dismiss();
  };

  const renderIcon = () => {
    switch (type) {
      case 'success':
        return (
          <div className="bg-green-500 w-7 h-7 rounded-full flex items-center justify-center shadow-lg shadow-green-500/20">
            <FaCheckCircle className="text-white text-base" />
          </div>
        );
      case 'error':
        return (
          <div className="bg-red-500 w-7 h-7 rounded-lg flex items-center justify-center shadow-lg shadow-red-500/20 transform rotate-45">
            <FaExclamationCircle className="text-white text-base transform -rotate-45" />
          </div>
        );
      case 'info':
        return (
          <div className="bg-blue-500 w-7 h-7 rounded-full flex items-center justify-center shadow-lg shadow-blue-500/20">
            <FaInfoCircle className="text-white text-base" />
          </div>
        );
      case 'warning':
        return (
          <div className="bg-amber-500 w-7 h-7 rounded-full flex items-center justify-center shadow-lg shadow-amber-500/20">
            <FaExclamationCircle className="text-white text-base" />
          </div>
        );
      case 'upload':
        return (
          <div className="bg-blue-500/10 p-2.5 rounded-xl border border-blue-500/20 text-blue-500">
            <FaCloudUploadAlt className="text-xl" />
          </div>
        );
      case 'update':
        return (
          <div className="bg-gray-900/5 p-2.5 rounded-xl border border-gray-200 text-blue-500">
            <FaCog className="text-xl animate-spin" style={{ animationDuration: '8s' }} />
          </div>
        );
      case 'loading':
        return (
          <div className="bg-blue-500/10 p-2.5 rounded-xl border border-blue-500/20 text-blue-500">
            <FaSpinner className="text-xl animate-spin" />
          </div>
        );
      case 'message':
        return (
          <div className="relative">
            <div className="w-11 h-11 rounded-full overflow-hidden border-2 border-white shadow-sm bg-gray-200/20">
              <img src={avatar || "https://i.pravatar.cc/150"} alt="" className="w-full h-full object-cover" />
            </div>
            <div className="absolute bottom-0 right-0 w-3 h-3 bg-green-500 border-2 border-white rounded-full"></div>
          </div>
        );
      default:
        return null;
    }
  };

  const isDarkAction = type === 'error' || type === 'warning';

  return (
    <div 
      className="bg-white/95 backdrop-blur-xl text-gray-900 p-3.5 rounded-2xl shadow-2xl flex flex-col gap-3 min-w-[320px] max-w-[420px] border border-gray-200 animate-in fade-in slide-in-from-bottom-4 duration-300 pointer-events-auto select-none"
      onClick={(e) => e.stopPropagation()}
    >
      <div className="flex items-start justify-between gap-3.5">
        <div className="flex items-center gap-3.5 flex-1 min-w-0">
          <div className="flex-shrink-0">
            {renderIcon()}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <h3 className="font-bold text-sm text-gray-900 leading-tight break-words">{title}</h3>
              {version && <span className="text-gray-900/40 text-[10px] font-mono whitespace-nowrap">| {version}</span>}
            </div>
            {description && (
              <p className="text-gray-900/60 text-xs mt-0.5 italic leading-relaxed break-words">
                {description}
              </p>
            )}
          </div>
        </div>
        <button 
          type="button"
          onPointerDown={(e) => {
            e.preventDefault();
            e.stopPropagation();
            dismiss();
          }} 
          className="text-gray-900/30 hover:text-gray-900 transition-colors flex-shrink-0 p-1.5 -m-1.5 cursor-pointer relative z-50"
        >
          <FaTimes size={14} />
        </button>
      </div>

      {(type === 'upload' || type === 'loading') && typeof progress === 'number' && (
        <div className="space-y-1.5 px-0.5">
          <div className="h-1.5 w-full bg-gray-200/30 rounded-full overflow-hidden">
            <div 
              className="h-full bg-blue-500 rounded-full transition-all duration-500" 
              style={{ width: `${progress}%` }}
            ></div>
          </div>
          <div className="flex justify-end">
            <span className="text-[10px] font-bold text-gray-900/40">{progress}%</span>
          </div>
        </div>
      )}

      {(actionLabel || type === 'message' || type === 'update' || type === 'upload' || type === 'loading') && (
        <div className={`flex items-center gap-2.5 ${type === 'message' ? 'ml-14' : (type === 'update' || type === 'upload' || type === 'loading') ? 'ml-[52px]' : ''}`}>
          <button 
            type="button"
            onPointerDown={(e) => {
              e.preventDefault();
              e.stopPropagation();
              dismiss();
            }}
            className="px-3.5 py-1.5 text-gray-900/70 hover:bg-gray-900/5 rounded-xl text-[11px] font-bold border border-gray-200 transition-all whitespace-nowrap cursor-pointer relative z-50"
          >
            {type === 'update' ? 'Skip' : type === 'upload' ? 'Cancel' : 'Dismiss'}
          </button>
          <button 
            type="button"
            onPointerDown={(e) => {
              e.preventDefault();
              e.stopPropagation();
              handleAction();
            }}
            className={`px-4 py-1.5 rounded-xl text-[11px] font-bold transition-all whitespace-nowrap shadow-sm cursor-pointer relative z-50 ${
              isDarkAction 
                ? 'bg-gray-900 text-gray-100 hover:opacity-90' 
                : 'bg-blue-500 text-white hover:opacity-90'
            }`}
          >
            {actionLabel || (type === 'message' ? 'Reply' : type === 'update' ? 'Install' : type === 'upload' ? 'Retry' : 'Confirm')}
          </button>
        </div>
      )}
    </div>
  );
};

export default PremiumToast;"""
    
    def _get_boilerplate_premium_toast_helper(self) -> str:
        """Get boilerplate premiumToast helper file content."""
        return """import { toast } from 'sonner';
import PremiumToast, {type ToastType } from './PremiumToast';

interface PremiumToastOptions {
  description?: string;
  actionLabel?: string;
  onAction?: () => void;
  avatar?: string;
  progress?: number;
  version?: string;
  duration?: number;
  id?: string | number;
  simulateProgress?: boolean;
}

export const premiumToast = (type: ToastType, title: string, options: PremiumToastOptions = {}) => {
  const toastId = options.id || Math.random().toString(36).substring(2, 9);

  return toast.custom((t) => (
    <PremiumToast 
      onClose={() => toast.dismiss(t)}
      type={type}
      title={title}
      description={options.description}
      actionLabel={options.actionLabel}
      onAction={options.onAction}
      avatar={options.avatar}
      progress={options.progress}
      version={options.version}
    />
  ), {
    duration: options.duration || 5000,
    id: toastId
  });
};

premiumToast.success = (title: string, options?: PremiumToastOptions) => premiumToast('success', title, options);
premiumToast.error = (title: string, options?: PremiumToastOptions) => premiumToast('error', title, options);
premiumToast.info = (title: string, options?: PremiumToastOptions) => premiumToast('info', title, options);
premiumToast.warning = (title: string, options?: PremiumToastOptions) => premiumToast('warning', title, options);
premiumToast.loading = (title: string, progress?: number, options?: PremiumToastOptions) => premiumToast('loading', title, { ...options, progress });
premiumToast.upload = (title: string, progress: number, options?: PremiumToastOptions) => premiumToast('upload', title, { ...options, progress });
premiumToast.update = (title: string, version: string, options?: PremiumToastOptions) => premiumToast('update', title, { ...options, version });
premiumToast.message = (title: string, description: string, avatar: string, options?: PremiumToastOptions) => 
  premiumToast('message', title, { ...options, description, avatar });

premiumToast.dismiss = (id?: string | number) => toast.dismiss(id);

premiumToast.promise = async <T,>(
  promiseOrFn: Promise<T> | ((setProgress: (p: number) => void) => Promise<T>),
  msgs: {
    loading: string;
    success: string | ((data: T) => string);
    error: string | ((error: Error | unknown) => string);
  },
  options: Omit<PremiumToastOptions, 'id'> = {}
) => {
  const id = Math.random().toString(36).substring(2, 9);
  let interval: any = null;
  
  const setProgress = (progress: number) => {
    premiumToast.loading(msgs.loading, progress, { ...options, id });
  };

  if (options.simulateProgress) {
    let currentProgress = 0;
    interval = setInterval(() => {
      currentProgress += Math.floor(Math.random() * 15) + 5;
      if (currentProgress >= 95) {
        if (interval) clearInterval(interval);
      } else {
        setProgress(currentProgress);
      }
    }, 400);
  } else {
    premiumToast.loading(msgs.loading, options.progress, { ...options, id });
  }

  try {
    const result = await (typeof promiseOrFn === 'function' ? promiseOrFn(setProgress) : promiseOrFn);
    
    if (interval) clearInterval(interval);
    
    const successMsg = typeof msgs.success === 'function' ? msgs.success(result) : msgs.success;
    premiumToast.success(successMsg, { ...options, id });
    
    return result;
  } catch (error) {
    if (interval) clearInterval(interval);
    
    const errorMsg = typeof msgs.error === 'function' ? msgs.error(error as Error) : msgs.error;
    premiumToast.error(errorMsg, { ...options, id });
    
    throw error;
  }
};"""
    
    def _write_boilerplate_if_missing(self, target_path: str, file_path: str, content: str) -> bool:
        """Write boilerplate file if it doesn't exist."""
        full_path = self._resolve_path(target_path, file_path)
        existing = self._read_file_content(full_path)
        if not existing:
            self._write_file(full_path, content)
            return True
        return False
    
    def _check_required_packages(self, target_path: str) -> Dict:
        """Check if required TanStack packages are installed in package.json.
        
        Returns dict with package flags and warnings.
        """
        result = {
            'has_react_query': False,
            'has_eslint_plugin': False,
            'has_sonner': False,
            'has_react_icons': False,
            'warnings': []
        }
        
        package_json_path = self._resolve_path(target_path, 'package.json')
        content = self._read_file_content(package_json_path)
        
        if not content:
            result['warnings'].append('package.json not found - cannot verify dependencies')
            return result
        
        try:
            import json
            package_data = json.loads(content)
            
            dependencies = package_data.get('dependencies', {})
            dev_dependencies = package_data.get('devDependencies', {})
            
            # Check for @tanstack/react-query
            if '@tanstack/react-query' in dependencies:
                result['has_react_query'] = True
            else:
                result['warnings'].append('Missing @tanstack/react-query in dependencies. Run: npm i @tanstack/react-query')
            
            # Check for @tanstack/eslint-plugin-query
            if '@tanstack/eslint-plugin-query' in dev_dependencies:
                result['has_eslint_plugin'] = True
            else:
                result['warnings'].append('Missing @tanstack/eslint-plugin-query in devDependencies. Run: npm i -D @tanstack/eslint-plugin-query')
            
            # Check for sonner (toast library)
            if 'sonner' in dependencies:
                result['has_sonner'] = True
            else:
                result['warnings'].append('Missing sonner in dependencies. Run: npm i sonner')
            
            # Check for react-icons
            if 'react-icons' in dependencies:
                result['has_react_icons'] = True
            else:
                result['warnings'].append('Missing react-icons in dependencies. Run: npm i react-icons')
                
        except Exception as e:
            result['warnings'].append(f'Error parsing package.json: {str(e)}')
        
        return result
    
    def _verify_imports(self, target_path: str, generated_files: List[Dict]) -> Dict:
        """Verify that imports in generated files reference existing files.
        
        Returns dict with 'valid_imports', 'missing_imports', and 'warnings'.
        """
        result = {
            'valid_imports': [],
            'missing_imports': [],
            'warnings': []
        }
        
        import re
        
        for file_info in generated_files:
            file_path = file_info.get('file', '')
            if not file_path or file_info.get('boilerplate'):
                continue
            
            full_path = self._resolve_path(target_path, file_path)
            content = self._read_file_content(full_path)
            
            if not content:
                continue
            
            # Extract import statements
            # Pattern: import ... from "..." or import ... from '...'
            import_pattern = r'import\s+(?:{[^}]+}|\*\s+as\s+\w+|\w+(?:\s*,\s*\w+)*)\s+from\s+["\']([^"\']+)["\']'
            matches = re.findall(import_pattern, content)
            
            for import_path in matches:
                # Skip external packages (node_modules)
                if import_path.startswith('@') or not import_path.startswith('.'):
                    continue
                
                # Resolve relative import to absolute path
                # Import is relative to the file's directory
                file_dir = full_path.rsplit('/', 1)[0] if '/' in full_path else target_path
                
                # Handle relative paths
                if import_path.startswith('./'):
                    resolved = file_dir + '/' + import_path[2:]
                elif import_path.startswith('../'):
                    parts = import_path.split('..')
                    resolved = file_dir
                    for _ in range(len(parts) - 1):
                        resolved = resolved.rsplit('/', 1)[0] if '/' in resolved else resolved
                    remaining = parts[-1].lstrip('/')
                    if remaining:
                        resolved += '/' + remaining
                else:
                    resolved = file_dir + '/' + import_path
                
                # Add .ts extension if missing
                if not resolved.endswith('.ts') and not resolved.endswith('.tsx'):
                    resolved += '.ts'
                
                # Check if file exists
                resolved_full = self._resolve_path(target_path, resolved)
                exists = self._read_file_content(resolved_full) is not None
                
                if exists:
                    result['valid_imports'].append({
                        'file': file_path,
                        'import': import_path,
                        'resolved': resolved
                    })
                else:
                    result['missing_imports'].append({
                        'file': file_path,
                        'import': import_path,
                        'resolved': resolved
                    })
                    result['warnings'].append(f"Missing import target: {import_path} in {file_path} (resolved to {resolved})")
        
        return result
    
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
        import sys
        path = raw_path.replace('\\', '/').strip()
        original_path = path
        
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
                if original_path != relative:
                    print(f"[TanStackAgent] Sanitized path: {original_path} -> {relative}", file=sys.stderr)
                return relative
        
        # If it's become an absolute Windows path after stripping, try to make it relative
        # e.g. if `path/to/` is stripped and we're left with `D:/Projects/...`
        if ':' in path or path.startswith('/'):
            # It's still an absolute path, return just the filename as fallback
            filename = path.split('/')[-1] if '/' in path else path
            if original_path != filename:
                print(f"[TanStackAgent] Fallback to filename: {original_path} -> {filename}", file=sys.stderr)
            return filename
        
        if original_path != path:
            print(f"[TanStackAgent] Path sanitized: {original_path} -> {path}", file=sys.stderr)
        
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
        package_check = {}
        if target_exists:
            existing_files = self._scan_existing_structure(target_path)
            package_check = self._check_required_packages(target_path)
        
        context_summary = self._build_context_summary(target_path, existing_files)
        
        # Retrieve memory context from past generations
        memory_context = self._retrieve_memory_context(task)
        
        # ── Step 1: Generate the project structure ──
        # Build context sections
        context_sections = []
        
        base_context = (
            f"Initialize a TanStack Query project structure for: {task}\n\n"
            f"Target directory: {target_path}\n"
            f"Framework: {framework}\n\n"
        )
        context_sections.append(base_context)
        
        if memory_context:
            context_sections.append(memory_context)
        
        if existing_files:
            context_sections.append(f"The target directory already exists. Here are the existing files:\n{context_summary}\n\nGenerate files that integrate with the existing project. Add or update files as needed.\n")
        else:
            context_sections.append("The target directory does not exist yet (or is empty). Create a fresh project structure.\n")
        
        if 'next' in framework.lower():
            context_sections.append(
                "\n- Use 'use client' directive for App Router components\n"
                "- Create files under app/ or src/ based on project convention\n"
                "- Provide a TanStack Query Provider wrapper for the layout\n"
            )
        elif 'react' in framework.lower():
            context_sections.append(
                "\n- Standard React SPA structure\n"
                "- Wrap app with QueryClientProvider in index.tsx or App.tsx\n"
            )
        
        # Instructions that must be preserved
        instructions = """
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
        
        # Build prompt with smart truncation
        prompt = self._build_prompt_with_truncation(instructions, context_sections)
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        generated = call_llm(messages)
        
        # ── Step 2: Write files to disk ──
        written_files = self._write_generated_files(target_path, generated)
        
        # ── Step 3: Write boilerplate files if missing (avoid LLM regeneration) ──
        boilerplate_written = []
        
        # Write apiClient.ts if missing (boilerplate - same across all projects)
        if self._write_boilerplate_if_missing(target_path, 'src/utils/apiClient.ts', self._get_boilerplate_api_client()):
            boilerplate_written.append('src/utils/apiClient.ts')
        
        # Write CustomError if missing (boilerplate - same across all projects)
        if self._write_boilerplate_if_missing(target_path, 'src/utils/error.ts', self._get_boilerplate_custom_error()):
            boilerplate_written.append('src/utils/error.ts')
        
        # Write PremiumToast component if missing (boilerplate - same across all projects)
        if self._write_boilerplate_if_missing(target_path, 'src/components/ui/feedback/PremiumToast.tsx', self._get_boilerplate_premium_toast()):
            boilerplate_written.append('src/components/ui/feedback/PremiumToast.tsx')
        
        # Write premiumToast helper if missing (boilerplate - same across all projects)
        if self._write_boilerplate_if_missing(target_path, 'src/components/ui/feedback/index.ts', self._get_boilerplate_premium_toast_helper()):
            boilerplate_written.append('src/components/ui/feedback/index.ts')
        
        # Write client.ts if missing (boilerplate - same across all projects)
        if self._write_boilerplate_if_missing(target_path, 'src/connections/client.ts', self._get_boilerplate_client()):
            boilerplate_written.append('src/connections/client.ts')
        
        # Note: ApiResponse, PaginatedResponse, and queryKeys are LLM-generated
        # as they vary by project/module requirements
        
        # Add boilerplate files to result
        if boilerplate_written:
            written_files.extend([
                {'file': f, 'boilerplate': True, 'status': 'written'} 
                for f in boilerplate_written
            ])
        
        # ── Step 4: Verify imports in generated files ──
        import_verification = self._verify_imports(target_path, written_files)
        
        result = {
            'status': 'initialized',
            'target_path': target_path,
            'framework': framework,
            'mode': 'project_initialization',
            'existing_files_scanned': len(existing_files),
            'files_written': written_files,
            'raw_generated_code': generated,
            'package_check': package_check,
            'import_verification': import_verification
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
        
        # Check required packages
        package_check = self._check_required_packages(target_path)
        
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
        
        # Convert endpoint_name to PascalCase for folder
        module_name = endpoint_name.replace('_', ' ').title().replace(' ', '')
        if not module_name:
            module_name = 'Api'
        
        # Build context sections
        context_sections = []
        
        base_context = (
            f"Ingest API endpoint and generate full typing stack for: {task}\n\n"
            f"Target directory: {target_path}\n"
            f"API Endpoint: {api_endpoint}\n"
            f"HTTP Method: {http_method}\n"
        )
        context_sections.append(base_context)
        
        if memory_context:
            context_sections.append(memory_context)
        
        if payload:
            context_sections.append(f"Request Payload:\n{payload}\n")
        if response_example:
            context_sections.append(f"Response Example (captured from real API):\n{response_example}\nIMPORTANT: Use the actual response structure above to generate accurate TypeScript types.\n")
        
        context_sections.append(f"Existing project context:\n{context_summary}\n")
        
        # Instructions that must be preserved
        instructions = f"""
Generate these 3 files with the `filepath:` format:

- src/types/{endpoint_name}.ts: TypeScript interfaces based on the actual API response
- src/connections/{module_name}/function.ts: Pure API functions using client
- src/connections/{module_name}/index.ts: TanStack Query hooks

IMPORTANT TYPE GENERATION RULES:
- Analyze the response_example carefully to create accurate TypeScript interfaces
- Handle nested objects, arrays, and optional fields correctly
- Include both success and error response types if available
- Use proper TypeScript generics for type safety
- Types MUST ONLY be defined in src/types/{endpoint_name}.ts
- DO NOT define types inline in function.ts - import them instead
- Use ApiResponse<T> wrapper for responses
- Use CustomError for error typing

IMPORTANT CLIENT USAGE:
- DO NOT generate client.ts - it's a shared file at src/connections/client.ts
- Import client from "../client" in function.ts: import {{ client }} from "../client"
- The import path MUST be exactly: import {{ client }} from "../client"
- All modules share the same client for common functionality (auth, error handling, etc.)

IMPORTANT FUNCTION.TS RULES:
- Import types from ../../types/{endpoint_name} (e.g., import {{ Product }} from "../../types/product")
- DO NOT define interfaces/types inline in function.ts
- Only export functions that use the imported types
- Use the imported types in function signatures and return types
- Add JSDoc comments for each function
- Helper functions for query params (buildQueryParams)
- NO @tanstack/react-query imports

IMPORTANT INDEX.TS RULES:
- Import {{ useQuery, useMutation, useQueryClient, type UseQueryOptions }} from '@tanstack/react-query'
- Import premiumToast from "../../components/ui/feedback"
- Import functions from ./function
- Import queryKeys from "../../constants/queryKeys"
- Import CustomError from "../../utils/error"
- Import types from "../../types"
- Hook naming: useXQuery (queries), useXMutation (mutations)
- Mutation pattern: onSuccess invalidates queries + premiumToast.success, onError premiumToast.error
- Use queryKeys.module.lists() for invalidation

```filepath:src/types/{endpoint_name}.ts
// TypeScript interfaces based on actual API response
// Define ALL types here - do not define them in function.ts
```
```filepath:src/connections/{module_name}/function.ts
// Pure API functions using client - NO @tanstack/react-query imports
// Import MUST be: import {{ client }} from "../client"
// Import types: import {{ TypeName }} from "../../types/{endpoint_name}"
// DO NOT define types inline - import them from types file
```
```filepath:src/connections/{module_name}/index.ts
// TanStack Query hooks with proper error handling and toast notifications
```
"""
        
        # Build prompt with smart truncation
        prompt = self._build_prompt_with_truncation(instructions, context_sections)
        
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
            'endpoint_name': endpoint_name,
            'package_check': package_check
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
        
        # Build context sections
        context_sections = []
        
        base_context = (
            f"Generate an index.ts file (TanStack Query hooks) for: {task}\n\n"
            f"Target: src/connections/{module_name}/index.ts\n"
            f"Hook type: {hook_type}\n\n"
        )
        context_sections.append(base_context)
        
        if functions_file:
            context_sections.append(f"Functions file content:\n{functions_file}\n")
        if types_file:
            context_sections.append(f"Types file content:\n{types_file}\n")
        if query_key_name:
            context_sections.append(f"Query key name to use: {query_key_name}\n")
        
        context_sections.append(f"Existing project context:\n{context_summary}\n")
        
        # Instructions that must be preserved
        instructions = f"""
IMPORTANT: This file must follow the Connection Structure Guide pattern:
- Import {{ useQuery, useMutation, useQueryClient, type UseQueryOptions }} from '@tanstack/react-query'
- Import {{ premiumToast }} from "../../components/ui/feedback"
- Import functions from ./function (e.g., getRequest, createRequest)
- Import {{ queryKeys }} from "../../constants/queryKeys"
- Import type {{ CustomError }} from "../../utils/error"
- Import types from ../../types/{module_key}
- Hook naming: use{module_name}Query (queries), useCreate{module_name}Mutation (creates), useUpdate{module_name}Mutation (updates), useDelete{module_name}Mutation (deletes)
- For queries: use queryKey: queryKeys.{module_key}.list(filters) and queryFn
- For mutations: use mutationFn, onSuccess (invalidateQueries + premiumToast.success), onError (premiumToast.error)
- Use queryClient.invalidateQueries({{ queryKey: queryKeys.{module_key}.lists() }}) for list invalidation
- Use queryClient.invalidateQueries({{ queryKey: queryKeys.{module_key}.detail(id) }}) for detail invalidation
- For error handling: check error.statusCode (e.g., skip toast for 403)
- For store integration: import from ../../store/authStore etc.
- Add JSDoc comments on every exported hook
- Use TypeScript generics for type safety throughout

Write the index.ts file using this format:
```filepath:src/connections/{module_name}/index.ts
// TanStack Query hooks - imports from @tanstack/react-query and ./function
```
"""
        
        # Build prompt with smart truncation
        prompt = self._build_prompt_with_truncation(instructions, context_sections)
        
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
        
        # Build context sections
        context_sections = []
        
        base_context = (
            f"Generate TypeScript type definitions for: {task}\n\n"
            f"Target directory: {target_path}/src/types/\n"
        )
        context_sections.append(base_context)
        
        if payload:
            context_sections.append(f"Request Payload/Schema:\n{payload}\n")
        if response_example:
            context_sections.append(f"Response Example:\n{response_example}\n")
        
        context_sections.append(f"Existing project context:\n{context_summary}\n")
        
        # Instructions that must be preserved
        instructions = f"""
IMPORTANT TYPE GENERATION RULES:
- Use ApiResponse<T> wrapper for responses
- Use CustomError for error typing
- Handle nested objects, arrays, and optional fields correctly
- Use proper TypeScript generics for type safety
- Define both request and response types
- Use PaginatedResponse<T> for list responses

Write the types file using this format (ONLY interfaces, NO @tanstack/react-query):
```filepath:src/types/{type_name}.ts
// TypeScript types only
```
"""
        
        # Build prompt with smart truncation
        prompt = self._build_prompt_with_truncation(instructions, context_sections)
        
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
        
        # Derive endpoint_name from module_name (lowercase)
        endpoint_name = module_name.lower()
        
        # Build context sections
        context_sections = []
        
        base_context = (
            f"Generate a function.ts file for: {task}\n\n"
            f"Target: src/connections/{module_name}/function.ts\n"
            f"API Endpoint: {api_endpoint}\n"
            f"HTTP Method: {http_method}\n\n"
        )
        context_sections.append(base_context)
        
        if types_import:
            context_sections.append(f"Types to use:\n{types_import}\n")
        
        context_sections.append(f"Existing project context:\n{context_summary}\n")
        
        # Instructions that must be preserved
        instructions = f"""
IMPORTANT: This file must follow the Connection Structure Guide pattern:
- Import {{ client }} from "../client" (SHARED client at src/connections/client.ts)
- Import types from ../../types/ or ../../types/{endpoint_name}
- Use client.get<T>(), client.post<T>(), client.put<T>(), client.patch<T>(), or client.delete<T>()
- Client options: includeAuth, isFormData, customHeaders, errorMessage, params, responseType, authType
- Use URLSearchParams for building query params on GET requests
- Add JSDoc comments on every exported function
- Helper functions for query params (buildQueryParams)
- Use isFormData: true for file uploads
- Use errorMessage option for user-friendly error messages
- Use responseType: "blob" for file downloads
- Do NOT import from @tanstack/react-query in this file
- DO NOT generate client.ts - it's a shared file at src/connections/client.ts
- Import path MUST be: import {{ client }} from "../client"
- DO NOT define types inline - import them from types file
- Use ApiResponse<T> wrapper for responses
- Use CustomError for error typing

Write the function.ts file using this format:
```filepath:src/connections/{module_name}/function.ts
// Pure API functions using client - NO @tanstack/react-query imports
```
"""
        
        # Build prompt with smart truncation
        prompt = self._build_prompt_with_truncation(instructions, context_sections)
        
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
