from typing import Dict, Optional
from api.lmstudio_client import call_llm


class FlutterAgent:
    """Specialized agent for Flutter development tasks."""
    
    def __init__(self):
        self.status = "idle"
        self.system_prompt = """
You are a Flutter Agent specialized in modern Flutter and Dart development.

Your capabilities:
- Write Flutter widgets using modern Dart
- Design Flutter UI components with Material Design or Cupertino
- Implement Flutter state management (Provider, Riverpod, Bloc, GetX)
- Handle Flutter navigation and routing
- Debug Flutter applications
- Suggest Flutter best practices and performance optimizations
- Integrate Flutter packages and plugins
- Implement platform-specific code (Android/iOS)
- Handle asynchronous operations with Future and Stream
- Use Flutter animations and transitions

Best Practices:
- Use stateless widgets when possible
- Implement proper state management solutions
- Use const constructors for performance
- Follow Flutter widget composition patterns
- Implement proper error handling and loading states
- Use responsive design principles
- Follow Dart naming conventions
- Use proper widget lifecycle management
- Implement platform-specific adaptations when needed
- Use effective async/await patterns

Always provide clean, well-commented, and production-ready Flutter code.
"""
    
    def execute(self, task_data: Dict) -> Dict:
        """Execute a Flutter task."""
        self.status = "working"
        task_type = task_data.get('type')
        
        try:
            if task_type == 'generate':
                result = self._generate_widget(task_data)
            elif task_type == 'refactor':
                result = self._refactor_widget(task_data)
            elif task_type == 'debug':
                result = self._debug_widget(task_data)
            elif task_type == 'explain':
                result = self._explain_widget(task_data)
            else:
                result = self._generate_widget(task_data)  # Default to generate
            
            self.status = "idle"
            return result
        except Exception as e:
            self.status = "error"
            return {'error': str(e)}
    
    def _generate_widget(self, task_data: Dict) -> Dict:
        """Generate a Flutter widget."""
        requirements = task_data.get('requirements', '')
        design_spec = task_data.get('design_spec', '')
        context = task_data.get('context', {})
        
        prompt = f"Generate a Flutter widget for: {requirements}"
        if design_spec:
            prompt += f"\n\nDesign specifications to follow:\n{design_spec}"
        if context:
            prompt += f"\n\nAdditional context: {context}"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt + "\n\nProvide complete, well-structured Flutter code with imports, widget, and styling."}
        ]
        
        code = call_llm(messages)
        
        return {
            'status': 'generated',
            'code': code,
            'requirements': requirements,
            'design_spec': design_spec
        }
    
    def _refactor_widget(self, task_data: Dict) -> Dict:
        """Refactor a Flutter widget."""
        widget_code = task_data.get('code', '')
        requirements = task_data.get('requirements', 'improve code quality and performance')
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Refactor this Flutter widget to {requirements}:\n\n{widget_code}\n\nProvide the refactored code and explain the improvements."}
        ]
        
        refactored = call_llm(messages)
        
        return {
            'status': 'refactored',
            'refactored_code': refactored
        }
    
    def _debug_widget(self, task_data: Dict) -> Dict:
        """Debug a Flutter widget."""
        widget_code = task_data.get('code', '')
        error = task_data.get('error', '')
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Debug this Flutter widget:\n\n{widget_code}\n\nError: {error}\n\nIdentify the issue and provide the fix."}
        ]
        
        debug_result = call_llm(messages)
        
        return {
            'status': 'debugged',
            'fix': debug_result
        }
    
    def _explain_widget(self, task_data: Dict) -> Dict:
        """Explain a Flutter widget."""
        widget_code = task_data.get('code', '')
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Explain this Flutter widget:\n\n{widget_code}\n\nProvide a clear explanation of what it does, how it works, and key patterns used."}
        ]
        
        explanation = call_llm(messages)
        
        return {
            'status': 'explained',
            'explanation': explanation
        }
    
    def get_status(self) -> str:
        """Get current agent status."""
        return self.status
