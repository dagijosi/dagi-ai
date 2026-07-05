from typing import Dict, Optional
from api.lmstudio_client import call_llm


class GeneralAgent:
    """General-purpose agent for tasks that don't fit specialized categories."""
    
    def __init__(self):
        self.status = "idle"
        self.system_prompt = """
You are a General Agent capable of handling a wide variety of tasks.

Your capabilities:
- Answer general questions
- Provide explanations on various topics
- Help with problem-solving
- Assist with research and information gathering
- Provide recommendations and suggestions
- Handle tasks that don't require specialized domain knowledge

Always provide clear, helpful, and accurate responses. If a task requires specialized expertise (like coding, security, database, etc.), suggest delegating to the appropriate specialized agent.
"""
    
    def execute(self, task_data: Dict) -> Dict:
        """Execute a general task."""
        self.status = "working"
        task_type = task_data.get('type')
        
        try:
            if task_type == 'answer':
                result = self._answer_question(task_data)
            elif task_type == 'explain':
                result = self._explain_topic(task_data)
            elif task_type == 'recommend':
                result = self._provide_recommendation(task_data)
            elif task_type == 'general':
                result = self._handle_general(task_data)
            else:
                result = {'error': f'Unknown task type: {task_type}'}
            
            self.status = "idle"
            return result
        except Exception as e:
            self.status = "error"
            return {'error': str(e)}
    
    def _answer_question(self, task_data: Dict) -> Dict:
        """Answer a general question."""
        question = task_data.get('question')
        context = task_data.get('context', '')
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Question: {question}\n\nContext: {context}\n\nProvide a clear and helpful answer."}
        ]
        
        answer = call_llm(messages)
        
        return {
            'status': 'answered',
            'question': question,
            'answer': answer
        }
    
    def _explain_topic(self, task_data: Dict) -> Dict:
        """Explain a topic or concept."""
        topic = task_data.get('topic')
        detail_level = task_data.get('detail_level', 'medium')
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Explain this topic: {topic}\n\nDetail level: {detail_level}\n\nProvide a clear explanation."}
        ]
        
        explanation = call_llm(messages)
        
        return {
            'status': 'explained',
            'topic': topic,
            'explanation': explanation
        }
    
    def _provide_recommendation(self, task_data: Dict) -> Dict:
        """Provide recommendations."""
        request = task_data.get('request')
        criteria = task_data.get('criteria', '')
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Recommendation request: {request}\n\nCriteria: {criteria}\n\nProvide helpful recommendations with reasoning."}
        ]
        
        recommendation = call_llm(messages)
        
        return {
            'status': 'recommended',
            'request': request,
            'recommendation': recommendation
        }
    
    def _handle_general(self, task_data: Dict) -> Dict:
        """Handle a general task."""
        task = task_data.get('task')
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Task: {task}\n\nHelp with this task to the best of your ability."}
        ]
        
        result = call_llm(messages)
        
        return {
            'status': 'completed',
            'task': task,
            'result': result
        }
    
    def get_status(self) -> str:
        """Get current agent status."""
        return self.status
