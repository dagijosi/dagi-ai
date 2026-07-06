from typing import Dict, Optional
from api.lmstudio_client import call_llm
from tools.tool_manager import ToolManager


class CodeAgent:
    """Specialized agent for code-related tasks."""
    
    def __init__(self, tool_manager: Optional[ToolManager] = None):
        self.status = "idle"
        self.tool_manager = tool_manager
        self.system_prompt = """
You are a Code Agent specialized in software development tasks.

Your capabilities:
- Read and analyze code files
- Write new code based on requirements
- Refactor existing code for better quality
- Debug and fix bugs
- Explain code functionality

Always provide clear, well-structured responses with code examples when appropriate.
"""
    
    def execute(self, task_data: Dict) -> Dict:
        """Execute a code-related task."""
        self.status = "working"
        task_type = task_data.get('type')
        
        try:
            if task_type == 'analyze':
                result = self._analyze_code(task_data)
            elif task_type == 'generate':
                result = self._generate_code(task_data)
            elif task_type == 'debug':
                result = self._debug_code(task_data)
            elif task_type == 'refactor':
                result = self._refactor_code(task_data)
            elif task_type == 'explain':
                result = self._explain_code(task_data)
            elif task_type == 'review':
                result = self._review_code(task_data)
            else:
                result = {'error': f'Unknown task type: {task_type}'}
            
            self.status = "idle"
            return result
        except Exception as e:
            self.status = "error"
            return {'error': str(e)}
    
    def _analyze_code(self, task_data: Dict) -> Dict:
        """Analyze code structure and dependencies."""
        file_path = task_data.get('file_path')
        code = task_data.get('code')
        question = task_data.get('question', '')
        
        # Get code content either from file or direct input
        if code:
            code_content = code
            source = "direct input"
        elif file_path:
            # Read the file using tool manager
            if self.tool_manager:
                file_result = self.tool_manager.execute_tool(
                    'file',
                    operation='read',
                    file_path=file_path
                )
            else:
                # Fallback to direct tool instantiation
                from tools.file_tool import FileTool
                file_tool = FileTool()
                file_result = file_tool.execute(operation='read', file_path=file_path)
            
            if 'error' in file_result or not file_result.get('success'):
                return file_result
            
            code_content = file_result.get('content', '')
            source = file_path
        else:
            return {'error': 'Either code or file_path must be provided for analysis'}
        
        # Use LLM to analyze
        user_prompt = f"Analyze this code from {source}:\n\n{code_content}\n\n"
        if question:
            user_prompt += f"Question: {question}\n\n"
        user_prompt += "Provide analysis of structure, complexity, dependencies, and improvement suggestions."
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        analysis = call_llm(messages)
        
        return {
            'status': 'analyzed',
            'source': source,
            'analysis': analysis
        }
    
    def _generate_code(self, task_data: Dict) -> Dict:
        """Generate code based on requirements."""
        # Use the new architecture context fields
        task = task_data.get('task') or task_data.get('goal') or task_data.get('requirements', '')
        language = task_data.get('language', 'python')
        context = task_data.get('context', '')
        design_spec = task_data.get('design_spec', '')
        previous_code = task_data.get('previous_code', '')
        framework = task_data.get('framework', '')
        use_design_spec = task_data.get('use_design_spec', False)
        previous_results = task_data.get('previous_results', [])
        
        # Build comprehensive prompt
        prompt = f"Generate {language} code for: {task}"
        
        # Add framework context
        if framework:
            prompt += f"\n\nFramework: {framework}"
            # Adjust language based on framework
            if framework == 'react':
                language = 'javascript/tsx'
            elif framework == 'flutter':
                language = 'dart'
        
        # Add design spec if available and requested
        if design_spec and use_design_spec:
            prompt += f"\n\nDesign specifications to follow:\n{design_spec}"
        
        # Add previous code context if available
        if previous_code:
            prompt += f"\n\nPrevious code to build upon:\n{previous_code}"
        
        # Add additional context
        if context:
            prompt += f"\n\nAdditional context: {context}"
        
        # Add previous results for context
        if previous_results:
            prompt += f"\n\nPrevious steps completed: {len(previous_results)}"
            for i, result in enumerate(previous_results):
                if result.get('success'):
                    prompt += f"\n  - Step {i+1}: {result.get('agent', 'unknown')} - {result.get('action', 'unknown')}"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt + "\n\nProvide complete, well-commented code that directly addresses the specific task requirements."}
        ]
        
        code = call_llm(messages)
        
        return {
            'status': 'generated',
            'code': code,
            'language': language
        }
    
    def _debug_code(self, task_data: Dict) -> Dict:
        """Debug code issues."""
        file_path = task_data.get('file_path')
        error_message = task_data.get('error')
        
        # Read the file
        file_result = self.file_tool.execute({
            'operation': 'read',
            'file_path': file_path
        })
        
        if 'error' in file_result:
            return file_result
        
        code_content = file_result['content']
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Debug this code: {file_path}\n\nError: {error_message}\n\nCode:\n{code_content}\n\nIdentify the bug, explain the issue, and provide the fix."}
        ]
        
        debug_result = call_llm(messages)
        
        return {
            'status': 'debugged',
            'fix': debug_result,
            'explanation': debug_result
        }
    
    def _refactor_code(self, task_data: Dict) -> Dict:
        """Refactor code for better quality."""
        file_path = task_data.get('file_path')
        
        # Read the file
        file_result = self.file_tool.execute({
            'operation': 'read',
            'file_path': file_path
        })
        
        if 'error' in file_result:
            return file_result
        
        code_content = file_result['content']
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Refactor this code for better quality, readability, and maintainability: {file_path}\n\n{code_content}\n\nProvide the refactored code and explain the changes made."}
        ]
        
        refactor_result = call_llm(messages)
        
        return {
            'status': 'refactored',
            'file': file_path,
            'refactored_code': refactor_result
        }
    
    def _explain_code(self, task_data: Dict) -> Dict:
        """Explain code functionality."""
        file_path = task_data.get('file_path')
        
        # Read the file
        file_result = self.file_tool.execute({
            'operation': 'read',
            'file_path': file_path
        })
        
        if 'error' in file_result:
            return file_result
        
        code_content = file_result['content']
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Explain this code file: {file_path}\n\n{code_content}\n\nProvide a clear explanation of what the code does, how it works, and key components."}
        ]
        
        explanation = call_llm(messages)
        
        return {
            'status': 'explained',
            'file': file_path,
            'explanation': explanation
        }
    
    def _review_code(self, task_data: Dict) -> Dict:
        """Review code for best practices and optimization."""
        code = task_data.get('code', '')
        context = task_data.get('context', {})
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Review this code for best practices, optimization, and potential improvements:\n\n{code}\n\nContext: {context}\n\nProvide detailed feedback and suggestions for improvement."}
        ]
        
        review = call_llm(messages)
        
        return {
            'status': 'reviewed',
            'review': review,
            'original_code': code
        }
    
    def get_status(self) -> str:
        """Get current agent status."""
        return self.status
