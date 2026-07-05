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
        return "Git operations: status, add, commit, push, pull, branch, checkout, log, diff, restore, history"
    
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
            elif operation == 'restore':
                result = self._git_restore(task_data)
            elif operation == 'history':
                result = self._git_history(task_data)
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
        """Show differences between commits, branches, or files."""
        target = task_data.get('target', None)
        source = task_data.get('source', None)
        file_path = task_data.get('file_path', None)
        staged = task_data.get('staged', False)
        
        args = ['diff']
        
        if staged:
            args.append('--staged')
        
        if source and target:
            args.append(f'{source}..{target}')
        elif target:
            args.append(target)
        
        if file_path:
            args.append(file_path)
        
        result = self._run_git_command(args)
        return {
            'operation': 'diff',
            'source': source,
            'target': target,
            'file_path': file_path,
            'staged': staged,
            'result': result
        }
    
    def _git_restore(self, task_data: Dict) -> Dict:
        """Restore working tree files or staged changes."""
        file_path = task_data.get('file_path', '.')
        staged = task_data.get('staged', False)
        source = task_data.get('source', None)
        
        args = ['restore']
        
        if staged:
            args.append('--staged')
        
        if source:
            args.append('--source')
            args.append(source)
        
        args.append(file_path)
        
        result = self._run_git_command(args)
        return {
            'operation': 'restore',
            'file_path': file_path,
            'staged': staged,
            'source': source,
            'result': result
        }
    
    def _git_history(self, task_data: Dict) -> Dict:
        """Get detailed commit history with changes."""
        limit = task_data.get('limit', 10)
        file_path = task_data.get('file_path', None)
        since = task_data.get('since', None)  # e.g., '1 week ago'
        
        args = ['log', '-n', str(limit), '--pretty=format:%H|%an|%ae|%ad|%s', '--date=iso']
        
        if file_path:
            args.append('--')
            args.append(file_path)
        
        if since:
            args.append(f'--since={since}')
        
        result = self._run_git_command(args)
        
        # Parse the output into structured data
        commits = []
        if result['status'] == 'success':
            for line in result['stdout'].strip().split('\n'):
                if line:
                    parts = line.split('|', 4)
                    if len(parts) == 5:
                        commits.append({
                            'hash': parts[0],
                            'author': parts[1],
                            'email': parts[2],
                            'date': parts[3],
                            'message': parts[4]
                        })
        
        return {
            'operation': 'history',
            'limit': limit,
            'file_path': file_path,
            'since': since,
            'commits': commits,
            'result': result
        }
