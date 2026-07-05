import os
from typing import Dict, List, Optional, Any
from pathlib import Path

from .base_tool import BaseTool


class FileTool(BaseTool):
    """Tool for file system operations."""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
    
    @property
    def name(self) -> str:
        return "file"
    
    @property
    def description(self) -> str:
        return "File system operations: read, write, list, delete, rename, copy, move, search files and directories"
    
    @property
    def permissions(self) -> list:
        return ["file_read", "file_write", "file_delete", "file_manage"]
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute a file system operation."""
        task_data = kwargs
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
            elif operation == 'rename':
                result = self._rename_file(task_data)
            elif operation == 'copy':
                result = self._copy_file(task_data)
            elif operation == 'move':
                result = self._move_file(task_data)
            elif operation == 'find':
                result = self._find_files(task_data)
            elif operation == 'read_multiple':
                result = self._read_multiple_files(task_data)
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
    
    def _rename_file(self, task_data: Dict) -> Dict:
        """Rename a file or directory."""
        file_path = task_data.get('file_path')
        new_name = task_data.get('new_name')
        full_path = self.base_path / file_path
        
        if not full_path.exists():
            return {'error': f'File not found: {file_path}'}
        
        new_path = full_path.parent / new_name
        full_path.rename(new_path)
        
        return {
            'status': 'renamed',
            'old_path': file_path,
            'new_path': str(new_path.relative_to(self.base_path))
        }
    
    def _copy_file(self, task_data: Dict) -> Dict:
        """Copy a file or directory."""
        file_path = task_data.get('file_path')
        destination = task_data.get('destination')
        full_path = self.base_path / file_path
        dest_path = self.base_path / destination
        
        if not full_path.exists():
            return {'error': f'File not found: {file_path}'}
        
        import shutil
        if full_path.is_dir():
            shutil.copytree(full_path, dest_path)
        else:
            shutil.copy2(full_path, dest_path)
        
        return {
            'status': 'copied',
            'source': file_path,
            'destination': destination
        }
    
    def _move_file(self, task_data: Dict) -> Dict:
        """Move a file or directory."""
        file_path = task_data.get('file_path')
        destination = task_data.get('destination')
        full_path = self.base_path / file_path
        dest_path = self.base_path / destination
        
        if not full_path.exists():
            return {'error': f'File not found: {file_path}'}
        
        import shutil
        shutil.move(str(full_path), str(dest_path))
        
        return {
            'status': 'moved',
            'source': file_path,
            'destination': destination
        }
    
    def _find_files(self, task_data: Dict) -> Dict:
        """Find files by name, extension, or pattern."""
        name = task_data.get('name')
        extension = task_data.get('extension')
        search_path = task_data.get('search_path', '.')
        full_path = self.base_path / search_path
        
        if not full_path.exists():
            return {'error': f'Search path not found: {search_path}'}
        
        matches = []
        for item in full_path.rglob('*'):
            if item.is_file():
                if name and name in item.name:
                    matches.append(str(item.relative_to(self.base_path)))
                elif extension and item.suffix == extension:
                    matches.append(str(item.relative_to(self.base_path)))
        
        return {
            'status': 'found',
            'name': name,
            'extension': extension,
            'search_path': search_path,
            'matches': matches
        }
    
    def _read_multiple_files(self, task_data: Dict) -> Dict:
        """Read multiple files at once."""
        file_paths = task_data.get('file_paths', [])
        results = {}
        
        for file_path in file_paths:
            full_path = self.base_path / file_path
            if full_path.exists() and full_path.is_file():
                with open(full_path, 'r', encoding='utf-8') as f:
                    results[file_path] = f.read()
            else:
                results[file_path] = f'Error: File not found'
        
        return {
            'status': 'read_multiple',
            'files_read': len(file_paths),
            'contents': results
        }
