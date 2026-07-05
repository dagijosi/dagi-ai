import subprocess
from typing import Dict, List, Optional


class TerminalTool:
    """Tool for executing terminal commands."""
    
    def __init__(self, working_dir: str = "."):
        self.working_dir = working_dir
    
    def execute(self, task_data: Dict) -> Dict:
        """Execute a terminal command."""
        command = task_data.get('command')
        timeout = task_data.get('timeout', 30)
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                'status': 'executed',
                'command': command,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        except subprocess.TimeoutExpired:
            return {
                'status': 'timeout',
                'command': command,
                'error': f'Command timed out after {timeout} seconds'
            }
        except Exception as e:
            return {
                'status': 'error',
                'command': command,
                'error': str(e)
            }
    
    def execute_interactive(self, task_data: Dict) -> Dict:
        """Execute an interactive command."""
        command = task_data.get('command')
        inputs = task_data.get('inputs', [])
        
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=self.working_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(
                input='\n'.join(inputs) if inputs else None,
                timeout=30
            )
            
            return {
                'status': 'executed',
                'command': command,
                'return_code': process.returncode,
                'stdout': stdout,
                'stderr': stderr
            }
        except Exception as e:
            return {
                'status': 'error',
                'command': command,
                'error': str(e)
            }
