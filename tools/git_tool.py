import subprocess
from typing import Dict, List, Optional, Any

from .base_tool import BaseTool


class GitTool(BaseTool):
    """Tool for Git operations."""
    
    def __init__(self, working_dir: str = "."):
        self.working_dir = working_dir
    
    @property
    def name(self) -> str:
        return "git"
    
    @property
    def description(self) -> str:
        return "Git operations: status, add, commit, push, pull, branch, checkout, log, diff"
    
    @property
    def permissions(self) -> list:
        return ["git_read", "git_write"]
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute a Git operation."""
        task_data = kwargs
        operation = task_data.get('operation')
        
        try:
            if operation == 'status':
                result = self._git_status()
            elif operation == 'add':
                result = self._git_add(task_data)
            elif operation == 'commit':
                result = self._git_commit(task_data)
            elif operation == 'push':
                result = self._git_push(task_data)
            elif operation == 'pull':
                result = self._git_pull()
            elif operation == 'branch':
                result = self._git_branch(task_data)
            elif operation == 'checkout':
                result = self._git_checkout(task_data)
            elif operation == 'log':
                result = self._git_log(task_data)
            elif operation == 'diff':
                result = self._git_diff(task_data)
            else:
                result = {'error': f'Unknown operation: {operation}'}
            
            return result
        except Exception as e:
            return {'error': str(e)}
    
    def _run_git_command(self, args: List[str]) -> Dict:
        """Run a Git command and return the result."""
        try:
            result = subprocess.run(
                ['git'] + args,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                check=True
            )
            
            return {
                'status': 'success',
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        except subprocess.CalledProcessError as e:
            return {
                'status': 'error',
                'return_code': e.returncode,
                'stdout': e.stdout,
                'stderr': e.stderr
            }
    
    def _git_status(self) -> Dict:
        """Get Git repository status."""
        result = self._run_git_command(['status', '--porcelain'])
        return {
            'operation': 'status',
            'result': result
        }
    
    def _git_add(self, task_data: Dict) -> Dict:
        """Add files to staging area."""
        files = task_data.get('files', '.')
        result = self._run_git_command(['add'] + [files] if isinstance(files, list) else [files])
        return {
            'operation': 'add',
            'files': files,
            'result': result
        }
    
    def _git_commit(self, task_data: Dict) -> Dict:
        """Commit staged changes."""
        message = task_data.get('message', 'Update')
        result = self._run_git_command(['commit', '-m', message])
        return {
            'operation': 'commit',
            'message': message,
            'result': result
        }
    
    def _git_push(self, task_data: Dict) -> Dict:
        """Push commits to remote."""
        remote = task_data.get('remote', 'origin')
        branch = task_data.get('branch', 'main')
        result = self._run_git_command(['push', remote, branch])
        return {
            'operation': 'push',
            'remote': remote,
            'branch': branch,
            'result': result
        }
    
    def _git_pull(self) -> Dict:
        """Pull changes from remote."""
        result = self._run_git_command(['pull'])
        return {
            'operation': 'pull',
            'result': result
        }
    
    def _git_branch(self, task_data: Dict) -> Dict:
        """List or create branches."""
        branch_name = task_data.get('branch_name')
        create_new = task_data.get('create_new', False)
        
        if create_new and branch_name:
            result = self._run_git_command(['checkout', '-b', branch_name])
        else:
            result = self._run_git_command(['branch', '-a'])
        
        return {
            'operation': 'branch',
            'branch_name': branch_name,
            'create_new': create_new,
            'result': result
        }
    
    def _git_checkout(self, task_data: Dict) -> Dict:
        """Checkout a branch or commit."""
        target = task_data.get('target')
        result = self._run_git_command(['checkout', target])
        return {
            'operation': 'checkout',
            'target': target,
            'result': result
        }
    
    def _git_log(self, task_data: Dict) -> Dict:
        """Get commit history."""
        limit = task_data.get('limit', 10)
        result = self._run_git_command(['log', '-n', str(limit), '--oneline'])
        return {
            'operation': 'log',
            'limit': limit,
            'result': result
        }
    
    def _git_diff(self, task_data: Dict) -> Dict:
        """Show differences between commits or files."""
        target = task_data.get('target', None)
        args = ['diff']
        if target:
            args.append(target)
        result = self._run_git_command(args)
        return {
            'operation': 'diff',
            'target': target,
            'result': result
        }
