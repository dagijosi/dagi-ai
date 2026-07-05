import os
from typing import Dict, List, Optional
from pathlib import Path


class FileTool:
    """Tool for file system operations."""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
    
    def execute(self, task_data: Dict) -> Dict:
        """Execute a file system operation."""
        operation = task_data.get('operation')
        
        try:
            if operation == 'read':
                result = self._read_file(task_data)
            elif operation == 'write':
                result = self._write_file(task_data)
            elif operation == 'list':
                result = self._list_directory(task_data)
            elif operation == 'delete':
                result = self._delete_file(task_data)
            elif operation == 'search':
                result = self._search_files(task_data)
            else:
                result = {'error': f'Unknown operation: {operation}'}
            
            return result
        except Exception as e:
            return {'error': str(e)}
    
    def _read_file(self, task_data: Dict) -> Dict:
        """Read file contents."""
        file_path = task_data.get('file_path')
        full_path = self.base_path / file_path
        
        if not full_path.exists():
            return {'error': f'File not found: {file_path}'}
        
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            'status': 'read',
            'file_path': file_path,
            'content': content
        }
    
    def _write_file(self, task_data: Dict) -> Dict:
        """Write content to file."""
        file_path = task_data.get('file_path')
        content = task_data.get('content')
        full_path = self.base_path / file_path
        
        # Create parent directories if they don't exist
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            'status': 'written',
            'file_path': file_path,
            'size': len(content)
        }
    
    def _list_directory(self, task_data: Dict) -> Dict:
        """List directory contents."""
        dir_path = task_data.get('dir_path', '.')
        full_path = self.base_path / dir_path
        
        if not full_path.exists():
            return {'error': f'Directory not found: {dir_path}'}
        
        items = []
        for item in full_path.iterdir():
            items.append({
                'name': item.name,
                'type': 'directory' if item.is_dir() else 'file',
                'size': item.stat().st_size if item.is_file() else 0
            })
        
        return {
            'status': 'listed',
            'dir_path': dir_path,
            'items': items
        }
    
    def _delete_file(self, task_data: Dict) -> Dict:
        """Delete a file or directory."""
        file_path = task_data.get('file_path')
        full_path = self.base_path / file_path
        
        if not full_path.exists():
            return {'error': f'File not found: {file_path}'}
        
        if full_path.is_dir():
            import shutil
            shutil.rmtree(full_path)
        else:
            full_path.unlink()
        
        return {
            'status': 'deleted',
            'file_path': file_path
        }
    
    def _search_files(self, task_data: Dict) -> Dict:
        """Search for files by pattern."""
        pattern = task_data.get('pattern', '*')
        search_path = task_data.get('search_path', '.')
        full_path = self.base_path / search_path
        
        if not full_path.exists():
            return {'error': f'Search path not found: {search_path}'}
        
        matches = list(full_path.rglob(pattern))
        
        return {
            'status': 'searched',
            'pattern': pattern,
            'search_path': search_path,
            'matches': [str(m.relative_to(self.base_path)) for m in matches]
        }
