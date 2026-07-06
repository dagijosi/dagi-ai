from typing import Dict, Any
from registry.agent_registry import BaseAgent


class AgentAdapter(BaseAgent):
    """Adapter to make existing agents compatible with BaseAgent interface."""
    
    def __init__(self, name: str, agent_instance):
        self.name = name
        self.agent = agent_instance
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent with given context."""
        try:
            # Convert context to task_data format expected by existing agents
            task_data = self._convert_context_to_task_data(context)
            
            # Execute the original agent
            result = self.agent.execute(task_data)
            
            # Ensure result is a dict
            if not isinstance(result, dict):
                return {
                    "success": True,
                    "result": result,
                    "agent": self.name
                }
            
            return result
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "agent": self.name
            }
    
    def get_capabilities(self) -> list:
        """Get capabilities based on agent type."""
        capability_map = {
            "code": ["read_code", "write_code", "analyze_code", "debug_code", "generate_code"],
            "memory": ["save_memory", "retrieve_memory", "update_memory", "search_memory"],
            "docs": ["read_docs", "search_docs", "answer_questions", "generate_docs"],
            "planner": ["create_plan", "validate_plan", "optimize_plan"],
            "general": ["answer", "explain", "recommend", "analyze"],
            "ui_designer": ["design_ui", "create_wireframes", "design_components"],
            "qa_tester": ["write_tests", "run_tests", "analyze_coverage", "debug_tests"],
            "security_auditor": ["audit_code", "check_vulnerabilities", "security_review"],
            "database_expert": ["design_schema", "write_queries", "optimize_db", "migrate_db"],
            "devops": ["deploy", "configure_ci", "monitor", "infrastructure"],
            "api": ["design_api", "implement_endpoints", "test_api", "document_api"],
            "flutter": ["create_flutter_components", "mobile_ui", "flutter_state"],
            "react": ["create_react_components", "web_ui", "react_hooks", "state_management"],
            "tanstack": ["initialize_tanstack", "ingest_api", "generate_types", "generate_hooks", "generate_service", "query_key_factory", "cache_management"]
        }
        
        return capability_map.get(self.name, ["general"])
    
    def _convert_context_to_task_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Convert new architecture context to existing agent task_data format."""
        task_data = {}
        
        # Map planner actions to agent task types
        action_to_type_map = {
            "generate": "generate",
            "read_and_analyze": "analyze",
            "find_relevant_files": "search",
            "generate_code": "generate",
            "read": "read",
            "write_files": "write",
            "save_context": "save",
            "design": "design",
            "test": "test",
            "analyze": "analyze",
            "answer": "answer",
            "filename": "search"
        }
        
        # Map common context fields to task_data fields
        if "task" in context:
            task_data["task"] = context["task"]
        if "description" in context:
            task_data["task"] = context["description"]
        if "action" in context:
            # Map action to type
            action = context["action"]
            task_data["type"] = action_to_type_map.get(action, action)
        if "goal" in context:
            task_data["goal"] = context["goal"]
            # Also set task to goal if no task specified
            if "task" not in task_data:
                task_data["task"] = context["goal"]
        if "type" in context:
            task_data["type"] = context["type"]
        if "operation" in context:
            task_data["operation"] = context["operation"]
        
        # Pass through design specs for code agents
        if "design_spec" in context:
            task_data["design_spec"] = context["design_spec"]
        
        # Pass through previous code for context
        if "previous_code" in context:
            task_data["previous_code"] = context["previous_code"]
        
        # Pass through previous results for context
        if "previous_results" in context:
            task_data["previous_results"] = context["previous_results"]
        
        # Add any additional context
        for key, value in context.items():
            if key not in task_data:
                task_data[key] = value
        
        # Ensure type field exists
        if "type" not in task_data:
            task_data["type"] = "general"
        
        return task_data
    
    def get_status(self) -> str:
        """Get agent status."""
        if hasattr(self.agent, "get_status"):
            return self.agent.get_status()
        return "idle"
