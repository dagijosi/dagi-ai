from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
from planner.planner import TaskGraph, TaskStep


class ExecutionResult:
    """Result of a single task step execution."""
    def __init__(self, step: int, success: bool, result: Any, error: Optional[str] = None):
        self.step = step
        self.success = success
        self.result = result
        self.error = error
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "step": self.step,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "timestamp": self.timestamp
        }


class Executor:
    """Executes task graphs created by the planner."""
    
    def __init__(self, agent_registry=None, tool_manager=None, event_bus=None):
        self.agent_registry = agent_registry
        self.tool_manager = tool_manager
        self.event_bus = event_bus
        self.execution_state = {}
        self.current_plan = None
    
    async def execute_plan(self, task_graph: TaskGraph) -> Dict[str, Any]:
        """Execute a complete task graph."""
        self.current_plan = task_graph
        results = []
        
        # Emit plan started event
        if self.event_bus:
            await self.event_bus.emit("plan_started", {
                "goal": task_graph.goal,
                "steps": len(task_graph.steps)
            })
        
        # Execute steps in order (respecting dependencies)
        for step in task_graph.steps:
            # Check if dependencies are satisfied
            if not self._check_dependencies(step, results):
                error_msg = f"Step {step.step} dependencies not satisfied"
                results.append(ExecutionResult(step.step, False, None, error_msg))
                continue
            
            # Execute the step
            result = await self._execute_step(step, results)
            results.append(result)
            
            # Emit step completed event
            if self.event_bus:
                await self.event_bus.emit("step_completed", {
                    "step": step.step,
                    "success": result.success,
                    "agent": step.agent
                })
            
            # If step failed, stop execution
            if not result.success:
                if self.event_bus:
                    await self.event_bus.emit("plan_failed", {
                        "goal": task_graph.goal,
                        "failed_step": step.step,
                        "error": result.error
                    })
                break
        
        # Emit plan completed event
        if self.event_bus:
            await self.event_bus.emit("plan_completed", {
                "goal": task_graph.goal,
                "total_steps": len(task_graph.steps),
                "successful_steps": sum(1 for r in results if r.success)
            })
        
        return {
            "goal": task_graph.goal,
            "results": [r.to_dict() for r in results],
            "success": all(r.success for r in results),
            "execution_state": self.execution_state
        }
    
    def _check_dependencies(self, step: TaskStep, results: List[ExecutionResult]) -> bool:
        """Check if step dependencies are satisfied."""
        if not step.dependencies:
            return True
        
        for dep_step in step.dependencies:
            # Find the result for the dependency step
            dep_result = next((r for r in results if r.step == dep_step), None)
            if not dep_result or not dep_result.success:
                return False
        
        return True
    
    async def _execute_step(self, step: TaskStep, previous_results: List[ExecutionResult]) -> ExecutionResult:
        """Execute a single task step."""
        try:
            # Build context for this step
            context = self._build_step_context(step, previous_results)
            
            # Execute based on agent type - distinguish between tools and agents
            tool_names = ['file', 'search', 'terminal', 'git', 'memory']
            
            if step.agent in tool_names:
                # These are tools, execute via tool manager
                if step.agent == 'file':
                    result = await self._execute_file_tool(step, context)
                elif step.agent == 'search':
                    result = await self._execute_search_tool(step, context)
                elif step.agent == 'terminal':
                    result = await self._execute_terminal_tool(step, context)
                elif step.agent == 'git':
                    result = await self._execute_git_tool(step, context)
                elif step.agent == 'memory':
                    result = await self._execute_memory_agent(step, context)
                else:
                    result = ExecutionResult(
                        step.step,
                        False,
                        None,
                        f"Unknown tool: {step.agent}"
                    )
            elif self.agent_registry and self.agent_registry.is_registered(step.agent):
                # These are registered agents
                result = await self._execute_agent(step, context)
            else:
                # Try to execute as agent anyway (might be in old agents dict)
                result = await self._execute_agent(step, context)
            
            # Update execution state
            self.execution_state[f"step_{step.step}"] = {
                "agent": step.agent,
                "action": step.action,
                "result": result.result if result.success else result.error,
                "timestamp": result.timestamp
            }
            
            return result
            
        except Exception as e:
            return ExecutionResult(step.step, False, None, str(e))
    
    def _build_step_context(self, step: TaskStep, previous_results: List[ExecutionResult]) -> Dict:
        """Build context for a step based on previous results."""
        context = {
            "step": step.step,
            "agent": step.agent,
            "action": step.action,
            "description": step.description,
            "goal": step.description  # Use description as goal for better context
        }
        
        # Add context requirements from the step
        context.update(step.context_requirements)
        
        # Add results from previous steps if needed
        if step.context_requirements.get("files_from_previous_step"):
            # Find file results from previous steps
            for result in previous_results:
                if result.success and isinstance(result.result, dict):
                    if "files" in result.result:
                        context["previous_files"] = result.result["files"]
        
        if step.context_requirements.get("code_from_previous_step"):
            # Find code results from previous steps and add them to context
            previous_code = []
            for result in previous_results:
                if result.success and isinstance(result.result, dict):
                    if "code" in result.result:
                        previous_code.append(result.result["code"])
                    if "design_spec" in result.result:
                        context["design_spec"] = result.result["design_spec"]
            if previous_code:
                context["previous_code"] = "\n\n".join(previous_code)
        
        # Add previous step results for context continuity
        if previous_results:
            context["previous_results"] = [
                {
                    "agent": r.agent if hasattr(r, 'agent') else step.agent,
                    "action": r.action if hasattr(r, 'action') else step.action,
                    "success": r.success,
                    "result": r.result if r.success else None
                }
                for r in previous_results
            ]
        
        return context
    
    async def _execute_file_tool(self, step: TaskStep, context: Dict) -> ExecutionResult:
        """Execute file tool operation."""
        if not self.tool_manager:
            return ExecutionResult(step.step, False, None, "Tool manager not available")
        
        try:
            result = self.tool_manager.execute_tool('file', **context)
            return ExecutionResult(step.step, True, result)
        except Exception as e:
            return ExecutionResult(step.step, False, None, str(e))
    
    async def _execute_search_tool(self, step: TaskStep, context: Dict) -> ExecutionResult:
        """Execute search tool operation."""
        if not self.tool_manager:
            return ExecutionResult(step.step, False, None, "Tool manager not available")
        
        try:
            result = self.tool_manager.execute_tool('search', **context)
            return ExecutionResult(step.step, True, result)
        except Exception as e:
            return ExecutionResult(step.step, False, None, str(e))
    
    async def _execute_terminal_tool(self, step: TaskStep, context: Dict) -> ExecutionResult:
        """Execute terminal tool operation."""
        if not self.tool_manager:
            return ExecutionResult(step.step, False, None, "Tool manager not available")
        
        try:
            result = self.tool_manager.execute_tool('terminal', **context)
            return ExecutionResult(step.step, True, result)
        except Exception as e:
            return ExecutionResult(step.step, False, None, str(e))
    
    async def _execute_git_tool(self, step: TaskStep, context: Dict) -> ExecutionResult:
        """Execute git tool operation."""
        if not self.tool_manager:
            return ExecutionResult(step.step, False, None, "Tool manager not available")
        
        try:
            result = self.tool_manager.execute_tool('git', **context)
            return ExecutionResult(step.step, True, result)
        except Exception as e:
            return ExecutionResult(step.step, False, None, str(e))
    
    async def _execute_memory_agent(self, step: TaskStep, context: Dict) -> ExecutionResult:
        """Execute memory agent operation."""
        if not self.tool_manager:
            return ExecutionResult(step.step, False, None, "Tool manager not available")
        
        try:
            result = self.tool_manager.execute_tool('memory', **context)
            return ExecutionResult(step.step, True, result)
        except Exception as e:
            return ExecutionResult(step.step, False, None, str(e))
    
    async def _execute_agent(self, step: TaskStep, context: Dict) -> ExecutionResult:
        """Execute agent operation."""
        if not self.agent_registry:
            return ExecutionResult(step.step, False, None, "Agent registry not available")
        
        try:
            agent = self.agent_registry.get_agent(step.agent)
            if not agent:
                return ExecutionResult(step.step, False, None, f"Agent not found: {step.agent}")
            
            # Execute agent with context
            result = agent.execute(context)
            
            # Check if execution was successful
            if isinstance(result, dict):
                success = result.get("success", True)  # Assume success if not specified
                if "error" in result:
                    success = False
            else:
                success = True
            
            return ExecutionResult(step.step, success, result)
        except Exception as e:
            return ExecutionResult(step.step, False, None, str(e))
    
    def get_execution_state(self) -> Dict:
        """Get current execution state."""
        return self.execution_state.copy()
    
    def reset_state(self):
        """Reset execution state."""
        self.execution_state = {}
        self.current_plan = None
