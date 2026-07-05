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
        return "Search operations: grep, regex search, and file finding"
    
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
