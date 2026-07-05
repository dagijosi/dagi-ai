import subprocess
from typing import Dict, List, Optional, Any

from .base_tool import BaseTool


class TerminalTool(BaseTool):
    """Tool for executing terminal commands."""
    
    def __init__(self, working_dir: str = "."):
        self.working_dir = working_dir
    
    @property
    def name(self) -> str:
        return "terminal"
    
    @property
    def description(self) -> str:
        return "Execute terminal commands with timeout and interactive support"
    
    @property
    def permissions(self) -> list:
        return ["terminal_execute"]
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute a terminal command."""
        task_data = kwargs
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
    
    def execute_interactive(self, **kwargs) -> Dict[str, Any]:
        """Execute an interactive command."""
        task_data = kwargs
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
