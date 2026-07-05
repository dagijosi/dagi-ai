from typing import Dict, Optional
from api.lmstudio_client import call_llm


class UIDesignerAgent:
    """Specialized agent for UI/UX design tasks."""
    
    def __init__(self):
        self.status = "idle"
        self.system_prompt = """
You are a UI Designer Agent specialized in modern user interface and user experience design.

Your capabilities:
- Design user interfaces with modern aesthetics
- Create wireframes and mockups descriptions
- Suggest UX improvements and best practices
- Design component libraries and design systems
- Provide color schemes, typography, and layout recommendations
- Ensure responsive and accessible design

Design Principles:
- Use modern, clean aesthetics
- Prioritize user experience and accessibility
- Ensure responsive design for all screen sizes
- Follow current design trends (minimalism, glassmorphism, neumorphism, etc.)
- Use appropriate color psychology and contrast
- Implement clear visual hierarchy
- Ensure intuitive navigation and interactions

Always provide detailed, actionable design specifications that developers can implement.
"""
    
    def execute(self, task_data: Dict) -> Dict:
        """Execute a UI design task."""
        self.status = "working"
        task_type = task_data.get('type')
        
        try:
            if task_type == 'design':
                result = self._design_ui(task_data)
            elif task_type == 'wireframe':
                result = self._create_wireframe(task_data)
            elif task_type == 'component':
                result = self._design_component(task_data)
            elif task_type == 'system':
                result = self._design_system(task_data)
            else:
                result = self._design_ui(task_data)  # Default to design
            
            self.status = "idle"
            return result
        except Exception as e:
            self.status = "error"
            return {'error': str(e)}
    
    def _design_ui(self, task_data: Dict) -> Dict:
        """Design a user interface."""
        requirements = task_data.get('requirements', '')
        context = task_data.get('context', {})
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Design a UI for: {requirements}\n\nContext: {context}\n\nProvide detailed design specifications including layout, colors, typography, spacing, and component structure."}
        ]
        
        design = call_llm(messages)
        
        return {
            'status': 'designed',
            'design_spec': design,
            'requirements': requirements
        }
    
    def _create_wireframe(self, task_data: Dict) -> Dict:
        """Create a wireframe description."""
        requirements = task_data.get('requirements', '')
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Create a wireframe for: {requirements}\n\nDescribe the layout structure, component placement, and user flow."}
        ]
        
        wireframe = call_llm(messages)
        
        return {
            'status': 'wireframed',
            'wireframe': wireframe
        }
    
    def _design_component(self, task_data: Dict) -> Dict:
        """Design a UI component."""
        component_type = task_data.get('component_type', '')
        requirements = task_data.get('requirements', '')
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Design a {component_type} component: {requirements}\n\nProvide detailed specifications including props, states, styling, and interactions."}
        ]
        
        component = call_llm(messages)
        
        return {
            'status': 'component_designed',
            'component_spec': component
        }
    
    def _design_system(self, task_data: Dict) -> Dict:
        """Design a design system."""
        requirements = task_data.get('requirements', '')
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Create a design system for: {requirements}\n\nDefine color palette, typography scale, spacing system, component library, and design tokens."}
        ]
        
        system = call_llm(messages)
        
        return {
            'status': 'system_designed',
            'design_system': system
        }
    
    def get_status(self) -> str:
        """Get current agent status."""
        return self.status
