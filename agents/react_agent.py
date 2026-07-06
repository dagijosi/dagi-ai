from typing import Dict, Optional
from api.lmstudio_client import call_llm


class ReactAgent:
    """Specialized agent for React development tasks."""
    
    def __init__(self):
        self.status = "idle"
        self.system_prompt = """
You are a React Agent specialized in modern React and JavaScript/TypeScript development.

Your capabilities:
- Write modern React components using functional components and hooks
- Implement React hooks (useState, useEffect, useContext, useReducer, useCallback, useMemo, useRef)
- Manage React state with hooks and context
- Design React architecture and component structure
- Debug React applications
- Suggest React best practices and performance optimizations
- Implement routing with React Router
- Handle forms and user input
- Integrate with APIs and data fetching
- Use modern React patterns (composition, custom hooks, etc.)

Best Practices:
- Use functional components with hooks
- Implement proper error boundaries
- Optimize performance with useMemo, useCallback, React.memo
- Use TypeScript for type safety when applicable
- Follow React naming conventions
- Implement proper prop types or TypeScript interfaces
- Use CSS modules, styled-components, or Tailwind CSS for styling
- Ensure accessibility (ARIA labels, semantic HTML)
- Implement proper loading and error states
- Use React Query or SWR for data fetching

Always provide clean, well-commented, and production-ready React code.
"""
    
    def execute(self, task_data: Dict) -> Dict:
        """Execute a React task."""
        self.status = "working"
        task_type = task_data.get('type')
        
        try:
            if task_type == 'generate':
                result = self._generate_component(task_data)
            elif task_type == 'refactor':
                result = self._refactor_component(task_data)
            elif task_type == 'debug':
                result = self._debug_component(task_data)
            elif task_type == 'explain':
                result = self._explain_component(task_data)
            else:
                result = self._generate_component(task_data)  # Default to generate
            
            self.status = "idle"
            return result
        except Exception as e:
            self.status = "error"
            return {'error': str(e)}
    
    def _generate_component(self, task_data: Dict) -> Dict:
        """Generate a React component."""
        # Use the new architecture context fields
        task = task_data.get('task') or task_data.get('goal') or task_data.get('requirements', '')
        design_spec = task_data.get('design_spec', '')
        previous_code = task_data.get('previous_code', '')
        framework = task_data.get('framework', 'react')
        previous_results = task_data.get('previous_results', [])
        
        # Build comprehensive prompt
        prompt = f"Generate a React component for: {task}"
        
        # Add design spec if available
        if design_spec:
            prompt += f"\n\nDesign specifications to follow:\n{design_spec}"
        
        # Add previous code context if available
        if previous_code:
            prompt += f"\n\nPrevious code to build upon:\n{previous_code}"
        
        # Add framework context
        if framework:
            prompt += f"\n\nFramework: {framework}"
        
        # Add previous results for context
        if previous_results:
            prompt += f"\n\nPrevious steps completed: {len(previous_results)}"
            for i, result in enumerate(previous_results):
                if result.get('success'):
                    prompt += f"\n  - Step {i+1}: {result.get('agent', 'unknown')} - {result.get('action', 'unknown')}"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt + "\n\nProvide complete, well-structured React code with imports, component, and styling. Make sure the component directly addresses the specific task requirements."}
        ]
        
        code = call_llm(messages)
        
        return {
            'status': 'generated',
            'code': code,
            'requirements': task,
            'design_spec': design_spec
        }
    
    def _refactor_component(self, task_data: Dict) -> Dict:
        """Refactor a React component."""
        component_code = task_data.get('code', '')
        requirements = task_data.get('requirements', 'improve code quality and performance')
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Refactor this React component to {requirements}:\n\n{component_code}\n\nProvide the refactored code and explain the improvements."}
        ]
        
        refactored = call_llm(messages)
        
        return {
            'status': 'refactored',
            'refactored_code': refactored
        }
    
    def _debug_component(self, task_data: Dict) -> Dict:
        """Debug a React component."""
        component_code = task_data.get('code', '')
        error = task_data.get('error', '')
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Debug this React component:\n\n{component_code}\n\nError: {error}\n\nIdentify the issue and provide the fix."}
        ]
        
        debug_result = call_llm(messages)
        
        return {
            'status': 'debugged',
            'fix': debug_result
        }
    
    def _explain_component(self, task_data: Dict) -> Dict:
        """Explain a React component."""
        component_code = task_data.get('code', '')
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Explain this React component:\n\n{component_code}\n\nProvide a clear explanation of what it does, how it works, and key patterns used."}
        ]
        
        explanation = call_llm(messages)
        
        return {
            'status': 'explained',
            'explanation': explanation
        }
    
    def get_status(self) -> str:
        """Get current agent status."""
        return self.status
