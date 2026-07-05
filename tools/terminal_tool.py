import subprocess
from typing import Dict, List, Optional, Any

from .base_tool import BaseTool


class TerminalTool(BaseTool):
    """Tool for executing terminal commands with command whitelisting."""
    
    # Whitelist of allowed commands (can be extended)
    ALLOWED_COMMANDS = {
        # Development commands
        'npm', 'npm install', 'npm run', 'npm test', 'npm start', 'npm build',
        'python', 'python3', 'pip', 'pip3', 'pytest', 'python -m',
        'node', 'npx',
        'yarn', 'yarn install', 'yarn run', 'yarn test', 'yarn start', 'yarn build',
        
        # Git commands
        'git', 'git status', 'git log', 'git diff', 'git branch', 'git checkout',
        'git add', 'git commit', 'git push', 'git pull', 'git clone',
        
        # File operations (safe ones)
        'ls', 'dir', 'cd', 'pwd',
        'cat', 'type', 'head', 'tail',
        'grep', 'find', 'locate',
        'cp', 'copy', 'mv', 'move', 'mkdir', 'md',
        
        # System info
        'echo', 'print', 'date', 'whoami', 'hostname',
        
        # Build tools
        'make', 'cmake', 'cargo', 'go', 'rustc',
        'gcc', 'g++', 'clang', 'clang++',
        
        # Docker (safe operations)
        'docker ps', 'docker images', 'docker build',
    }
    
    # Blacklist of dangerous commands (always blocked)
    BLOCKED_COMMANDS = {
        'rm -rf', 'del', 'format', 'diskpart',
        'shutdown', 'reboot', 'halt', 'poweroff',
        'dd', 'mkfs', 'fdisk', 'parted',
        'chmod 777', 'chown', 'chmod -R',
        'sudo rm', 'sudo del', 'sudo format',
        '> /dev/sd', '> /dev/hd',
        ':(){:|:&};:',  # Fork bomb
    }
    
    def __init__(self, working_dir: str = ".", strict_mode: bool = True):
        self.working_dir = working_dir
        self.strict_mode = strict_mode  # If True, only allow whitelisted commands
    
    @property
    def name(self) -> str:
        return "terminal"
    
    @property
    def description(self) -> str:
        return "Execute terminal commands with timeout, interactive support, and command whitelisting for security"
    
    @property
    def permissions(self) -> list:
        return ["terminal_execute", "terminal_restricted"]
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute a terminal command with security checks."""
        task_data = kwargs
        command = task_data.get('command')
        timeout = task_data.get('timeout', 30)
        
        # Security check
        security_result = self._check_command_security(command)
        if not security_result['allowed']:
            return {
                'status': 'blocked',
                'command': command,
                'error': security_result['reason']
            }
        
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
        """Execute an interactive command with security checks."""
        task_data = kwargs
        command = task_data.get('command')
        inputs = task_data.get('inputs', [])
        
        # Security check
        security_result = self._check_command_security(command)
        if not security_result['allowed']:
            return {
                'status': 'blocked',
                'command': command,
                'error': security_result['reason']
            }
        
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
    
    def _check_command_security(self, command: str) -> Dict[str, Any]:
        """Check if a command is allowed to execute."""
        command_lower = command.lower().strip()
        
        # Check against blacklist first (always blocked)
        for blocked in self.BLOCKED_COMMANDS:
            if blocked.lower() in command_lower:
                return {
                    'allowed': False,
                    'reason': f'Command contains blocked pattern: {blocked}'
                }
        
        # If strict mode is enabled, check against whitelist
        if self.strict_mode:
            # Extract the base command (first word or common patterns)
            base_command = command_lower.split()[0] if command_lower.split() else ''
            
            # Check if command starts with any allowed command
            allowed = False
            for allowed_cmd in self.ALLOWED_COMMANDS:
                if command_lower.startswith(allowed_cmd.lower()):
                    allowed = True
                    break
            
            if not allowed:
                return {
                    'allowed': False,
                    'reason': f'Command not in whitelist: {base_command}. Use strict_mode=False to allow all non-blocked commands.'
                }
        
        return {'allowed': True}
    
    def add_allowed_command(self, command: str):
        """Add a command to the whitelist."""
        self.ALLOWED_COMMANDS.add(command)
    
    def add_blocked_command(self, command: str):
        """Add a command to the blacklist."""
        self.BLOCKED_COMMANDS.add(command)
    
    def get_whitelist(self) -> set:
        """Get the current whitelist."""
        return self.ALLOWED_COMMANDS.copy()
    
    def get_blacklist(self) -> set:
        """Get the current blacklist."""
        return self.BLOCKED_COMMANDS.copy()
