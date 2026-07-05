import re
from typing import Dict, List, Optional, Any
from pathlib import Path

from .base_tool import BaseTool


class SearchTool(BaseTool):
    """Tool for searching within files."""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
    
    @property
    def name(self) -> str:
        return "search"
    
    @property
    def description(self) -> str:
        return "Search operations: filename, extension, project, regex, grep, and semantic search"
    
    @property
    def permissions(self) -> list:
        return ["file_read"]
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute a search operation."""
        task_data = kwargs
        operation = task_data.get('operation')
        
        try:
            if operation == 'grep':
                result = self._grep_search(task_data)
            elif operation == 'regex':
                result = self._regex_search(task_data)
            elif operation == 'find':
                result = self._find_files(task_data)
            elif operation == 'filename':
                result = self._search_by_filename(task_data)
            elif operation == 'extension':
                result = self._search_by_extension(task_data)
            elif operation == 'project':
                result = self._search_by_project(task_data)
            elif operation == 'semantic':
                result = self._semantic_search(task_data)
            else:
                result = {'error': f'Unknown operation: {operation}'}
            
            return result
        except Exception as e:
            return {'error': str(e)}
    
    def _grep_search(self, task_data: Dict) -> Dict:
        """Search for text in files."""
        pattern = task_data.get('pattern')
        file_pattern = task_data.get('file_pattern', '*')
        case_sensitive = task_data.get('case_sensitive', False)
        
        results = []
        flags = 0 if case_sensitive else re.IGNORECASE
        
        for file_path in self.base_path.rglob(file_pattern):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            if re.search(pattern, line, flags):
                                results.append({
                                    'file': str(file_path.relative_to(self.base_path)),
                                    'line_number': line_num,
                                    'line': line.strip(),
                                    'match': re.search(pattern, line, flags).group()
                                })
                except Exception:
                    continue
        
        return {
            'status': 'searched',
            'pattern': pattern,
            'matches': results,
            'count': len(results)
        }
    
    def _regex_search(self, task_data: Dict) -> Dict:
        """Perform regex search with capture groups."""
        pattern = task_data.get('pattern')
        file_path = task_data.get('file_path')
        
        full_path = self.base_path / file_path
        
        if not full_path.exists():
            return {'error': f'File not found: {file_path}'}
        
        results = []
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                match = re.search(pattern, line)
                if match:
                    results.append({
                        'line_number': line_num,
                        'line': line.strip(),
                        'groups': match.groups(),
                        'groupdict': match.groupdict()
                    })
        
        return {
            'status': 'searched',
            'pattern': pattern,
            'file_path': file_path,
            'matches': results
        }
    
    def _find_files(self, task_data: Dict) -> Dict:
        """Find files by name pattern."""
        pattern = task_data.get('pattern', '*')
        search_path = task_data.get('search_path', '.')
        full_path = self.base_path / search_path
        
        if not full_path.exists():
            return {'error': f'Search path not found: {search_path}'}
        
        matches = list(full_path.rglob(pattern))
        
        return {
            'status': 'found',
            'pattern': pattern,
            'search_path': search_path,
            'files': [str(m.relative_to(self.base_path)) for m in matches if m.is_file()],
            'directories': [str(m.relative_to(self.base_path)) for m in matches if m.is_dir()]
        }
    
    def _search_by_filename(self, task_data: Dict) -> Dict:
        """Search for files by exact or partial filename."""
        filename = task_data.get('filename')
        search_path = task_data.get('search_path', '.')
        exact_match = task_data.get('exact_match', False)
        full_path = self.base_path / search_path
        
        if not full_path.exists():
            return {'error': f'Search path not found: {search_path}'}
        
        matches = []
        for item in full_path.rglob('*'):
            if item.is_file():
                if exact_match:
                    if item.name == filename:
                        matches.append(str(item.relative_to(self.base_path)))
                else:
                    if filename.lower() in item.name.lower():
                        matches.append(str(item.relative_to(self.base_path)))
        
        return {
            'status': 'found',
            'filename': filename,
            'exact_match': exact_match,
            'search_path': search_path,
            'matches': matches
        }
    
    def _search_by_extension(self, task_data: Dict) -> Dict:
        """Search for files by extension."""
        extension = task_data.get('extension')
        search_path = task_data.get('search_path', '.')
        full_path = self.base_path / search_path
        
        if not extension.startswith('.'):
            extension = '.' + extension
        
        if not full_path.exists():
            return {'error': f'Search path not found: {search_path}'}
        
        matches = []
        for item in full_path.rglob('*'):
            if item.is_file() and item.suffix == extension:
                matches.append(str(item.relative_to(self.base_path)))
        
        return {
            'status': 'found',
            'extension': extension,
            'search_path': search_path,
            'matches': matches
        }
    
    def _search_by_project(self, task_data: Dict) -> Dict:
        """Search for files within a specific project directory."""
        project_name = task_data.get('project_name')
        full_path = self.base_path
        
        project_path = None
        for item in full_path.iterdir():
            if item.is_dir() and project_name.lower() in item.name.lower():
                project_path = item
                break
        
        if not project_path:
            return {'error': f'Project not found: {project_name}'}
        
        matches = []
        for item in project_path.rglob('*'):
            if item.is_file():
                matches.append(str(item.relative_to(self.base_path)))
        
        return {
            'status': 'found',
            'project_name': project_name,
            'project_path': str(project_path.relative_to(self.base_path)),
            'matches': matches
        }
    
    def _semantic_search(self, task_data: Dict) -> Dict:
        """Perform semantic search using embeddings (placeholder for future implementation)."""
        query = task_data.get('query')
        search_path = task_data.get('search_path', '.')
        
        # This is a placeholder for semantic search using embeddings
        # In a full implementation, this would:
        # 1. Generate embeddings for the query
        # 2. Compare with pre-computed file embeddings
        # 3. Return most semantically similar files
        
        return {
            'status': 'not_implemented',
            'message': 'Semantic search requires embedding model integration',
            'query': query,
            'search_path': search_path
        }
