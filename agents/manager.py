from typing import Dict, List, Optional
from api.lmstudio_client import call_llm
from tools.tool_manager import ToolManager
from tools.file_tool import FileTool
from tools.terminal_tool import TerminalTool
from tools.search_tool import SearchTool
from tools.git_tool import GitTool
from tools.memory_tool import MemoryTool
from .code_agent import CodeAgent
from .memory_agent import MemoryAgent
from .docs_agent import DocsAgent
from .planner_agent import PlannerAgent
from .general_agent import GeneralAgent
from .ui_designer_agent import UIDesignerAgent
from .qa_tester_agent import QATesterAgent
from .security_auditor_agent import SecurityAuditorAgent
from .database_expert_agent import DatabaseExpertAgent
from .devops_agent import DevOpsAgent
from .api_agent import APIAgent
from .flutter_agent import FlutterAgent
from .react_agent import ReactAgent

# New architecture imports
from planner.planner import Planner
from executor.executor import Executor
from context.manager import ContextManager, ContextRequest
from state.shared_state import SharedState
from events.event_bus import EventBus, get_global_event_bus
from registry.agent_registry import AgentRegistry, BaseAgent
from system_logging.logger import SystemLogger, get_global_logger


class ManagerAgent:
    """Manager agent that coordinates between specialized agents using LLM."""
    
    def __init__(self):
        self.system_prompt = """
You are the Manager Agent in a sophisticated multi-agent AI system. Your role is to coordinate specialized agents to solve complex tasks by breaking them down and orchestrating multiple agents.

Available Agents:
- Code Agent: Analyzes, generates, debugs, refactors, and explains code
- Memory Agent: Manages conversation history, document embeddings, and corrections using ChromaDB
- Docs Agent: Reads PDFs, Markdown, Word documents; generates documentation; answers documentation questions
- Planner Agent: Creates and validates multi-step execution plans
- General Agent: Handles general questions, explanations, recommendations, and non-specialized tasks
- UI Designer Agent: UI/UX design, wireframes, mockups, component design
- QA Tester Agent: Unit tests, integration tests, test coverage
- Security Auditor Agent: Security analysis, vulnerability assessment
- Database Expert Agent: Database design, SQL queries, optimization
- DevOps Agent: CI/CD pipelines, infrastructure, deployment
- API Agent: REST/GraphQL API design and development
- Flutter Agent: Flutter and Dart development
- React Agent: React and JavaScript/TypeScript development

Your Responsibilities:
1. Analyze user requests to understand intent and complexity
2. Break down complex tasks into coordinated sub-tasks across multiple agents
3. Delegate sub-tasks to appropriate specialized agents in the right order
4. Coordinate agent execution, handling dependencies between agents
5. Aggregate and synthesize results from multiple agents into a cohesive response
6. Ensure the final output is modern, well-structured, and production-ready

Multi-Agent Coordination Examples:
- "Write a React login page" → UI Designer (design) → React Agent (implement) → Code Agent (review/optimize)
- "Create a REST API with documentation" → API Agent (design API) → Code Agent (implement) → Docs Agent (generate docs)
- "Build a secure authentication system" → Security Auditor (review requirements) → Database Expert (design schema) → Code Agent (implement) → QA Tester (test)

Decision Framework:
- For UI/UX design tasks → Use UI Designer Agent first, then framework-specific agent
- For framework-specific tasks (React, Flutter) → Use framework agent, potentially with UI Designer for design
- For code generation → Use Code Agent, potentially with framework agent for best practices
- For documentation tasks → Use Docs Agent  
- For memory retrieval/storage → Use Memory Agent
- For complex multi-step tasks → Break into steps using multiple agents
- For general questions, explanations, recommendations → Use General Agent
- For testing tasks → Use QA Tester Agent
- For security tasks → Use Security Auditor Agent
- For database tasks → Use Database Expert Agent
- For DevOps tasks → Use DevOps Agent
- For API tasks → Use API Agent

Response Format:
- State your understanding of the request
- Explain your multi-agent approach/plan
- Execute delegated tasks in order
- Aggregate and synthesize results into a cohesive response
- Provide the final integrated solution
- If uncertain, ask clarifying questions

Always think step-by-step, coordinate agents effectively, and ensure the final result is cohesive and high-quality.
"""
        
        # Initialize Tool Manager
        self.tool_manager = ToolManager()
        
        # Register tools
        self.tool_manager.register_tool(FileTool())
        self.tool_manager.register_tool(TerminalTool())
        self.tool_manager.register_tool(SearchTool())
        self.tool_manager.register_tool(GitTool())
        
        # Initialize agents with tool manager
        self.memory_agent = MemoryAgent()
        self.tool_manager.register_tool(MemoryTool(memory_agent=self.memory_agent))
        
        self.code_agent = CodeAgent(tool_manager=self.tool_manager)
        self.docs_agent = DocsAgent()
        self.planner_agent = PlannerAgent()
        self.general_agent = GeneralAgent()
        
        # Phase 4 Future Agents (stubs)
        self.ui_designer_agent = UIDesignerAgent()
        self.qa_tester_agent = QATesterAgent()
        self.security_auditor_agent = SecurityAuditorAgent()
        self.database_expert_agent = DatabaseExpertAgent()
        self.devops_agent = DevOpsAgent()
        self.api_agent = APIAgent()
        self.flutter_agent = FlutterAgent()
        self.react_agent = ReactAgent()
        
        # New Architecture Components
        self.shared_state = SharedState()
        self.event_bus = get_global_event_bus()
        self.logger = get_global_logger()
        self.context_manager = ContextManager(
            memory_agent=self.memory_agent,
            file_tool=self.tool_manager.get_tool('file'),
            search_tool=self.tool_manager.get_tool('search')
        )
        self.planner = Planner(llm_client=None)
        self.executor = Executor(
            agent_registry=None,  # Will be set up below
            tool_manager=self.tool_manager,
            event_bus=self.event_bus
        )
        self.agent_registry = AgentRegistry()
        
        self.agents = {
            'code': self.code_agent,
            'memory': self.memory_agent,
            'docs': self.docs_agent,
            'planner': self.planner_agent,
            'general': self.general_agent,
            'ui_designer': self.ui_designer_agent,
            'qa_tester': self.qa_tester_agent,
            'security_auditor': self.security_auditor_agent,
            'database_expert': self.database_expert_agent,
            'devops': self.devops_agent,
            'api': self.api_agent,
            'flutter': self.flutter_agent,
            'react': self.react_agent,
        }
    
    def run(self, user_input: str) -> Dict:
        """Process user input through the manager agent with multi-agent coordination."""
        # First, analyze the request and create an execution plan
        planning_prompt = f"""
{self.system_prompt}

User request: {user_input}

Analyze this request and create an execution plan. Break it down into steps that require different agents.

Respond in this exact JSON format:
{{
  "analysis": "brief analysis of what the user wants",
  "complexity": "simple|medium|complex",
  "plan": [
    {{
      "step": 1,
      "agent": "agent_name",
      "task": "specific task for this agent",
      "task_type": "appropriate task type for this agent"
    }}
  ]
}}

Available agents: code, memory, docs, planner, general, ui_designer, qa_tester, security_auditor, database_expert, devops, api, flutter, react

For simple requests, use a single step. For complex requests, break into multiple steps using different agents.
"""
        
        messages = [
            {"role": "system", "content": "You are a planning agent. Always respond with valid JSON."},
            {"role": "user", "content": planning_prompt}
        ]
        
        try:
            response = call_llm(messages)
            
            # Try to parse JSON from response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                plan_decision = json.loads(json_str)
                
                analysis = plan_decision.get('analysis', '')
                complexity = plan_decision.get('complexity', 'simple')
                plan_steps = plan_decision.get('plan', [])
                
                # Execute the plan with context passing between agents
                execution_results = []
                context = {}
                
                for step in plan_steps:
                    agent_name = step.get('agent')
                    task_description = step.get('task', '')
                    task_type = step.get('task_type', self._infer_task_type(agent_name, task_description))
                    
                    # Normalize task_type to handle variations like "Code review/optimize"
                    if agent_name == 'code':
                        task_type_lower = task_type.lower()
                        if 'review' in task_type_lower or 'optimize' in task_type_lower or 'best practices' in task_type_lower:
                            task_type = 'review'
                    
                    # Build task data with context from previous steps
                    task_data = {
                        'type': task_type,
                        'task': task_description,
                        'user_input': user_input,
                        'context': context  # Pass results from previous agents
                    }
                    
                    # Add specific fields based on agent
                    if agent_name == 'code':
                        task_data['requirements'] = task_description
                        if 'ui_design' in context:
                            task_data['design_spec'] = context['ui_design']
                    elif agent_name == 'docs':
                        task_data['question'] = task_description
                        if 'code' in context:
                            task_data['code_context'] = context['code']
                    elif agent_name == 'general':
                        task_data['question'] = task_description
                    elif agent_name == 'react':
                        task_data['requirements'] = task_description
                        if 'ui_design' in context:
                            task_data['design_spec'] = context['ui_design']
                    elif agent_name == 'flutter':
                        task_data['requirements'] = task_description
                        if 'ui_design' in context:
                            task_data['design_spec'] = context['ui_design']
                    elif agent_name == 'ui_designer':
                        task_data['requirements'] = task_description
                    
                    # Execute the step
                    try:
                        result = self.route_task(agent_name, task_data)
                        
                        # Store result in context for next agents
                        context[agent_name] = result
                        
                        execution_results.append({
                            'step': step.get('step', len(execution_results) + 1),
                            'agent': agent_name,
                            'task': task_description,
                            'status': 'success',
                            'result': result
                        })
                    except Exception as e:
                        execution_results.append({
                            'step': step.get('step', len(execution_results) + 1),
                            'agent': agent_name,
                            'task': task_description,
                            'status': 'error',
                            'error': str(e)
                        })
                
                # Synthesize final product from all agent results
                final_product = self._synthesize_final_product(user_input, execution_results, context)
                
                return {
                    'manager_analysis': {
                        'analysis': analysis,
                        'complexity': complexity,
                        'total_steps': len(plan_steps)
                    },
                    'execution_plan': plan_steps,
                    'execution_results': execution_results,
                    'final_product': final_product
                }
            else:
                # If JSON parsing fails, use general agent as fallback
                return {
                    'manager_analysis': {
                        'analysis': 'Could not parse plan, using general agent',
                        'complexity': 'simple',
                        'total_steps': 1
                    },
                    'execution_plan': [{'agent': 'general', 'task': user_input}],
                    'execution_results': [{
                        'step': 1,
                        'agent': 'general',
                        'task': user_input,
                        'status': 'success',
                        'result': self.route_task('general', {'type': 'general', 'task': user_input})
                    }]
                }
        except Exception as e:
            # On error, use general agent as fallback
            return {
                'manager_analysis': {
                    'analysis': f'Error in planning: {str(e)}, using general agent',
                    'complexity': 'simple',
                    'total_steps': 1
                },
                'execution_plan': [{'agent': 'general', 'task': user_input}],
                'execution_results': [{
                    'step': 1,
                    'agent': 'general',
                    'task': user_input,
                    'status': 'success',
                    'result': self.route_task('general', {'type': 'general', 'task': user_input})
                }]
            }
    
    def _infer_task_type(self, agent_name: str, user_input: str) -> str:
        """Infer the appropriate task type based on agent and input."""
        task_type_map = {
            'code': 'analyze',
            'memory': 'store_conversation',
            'docs': 'answer_question',
            'planner': 'create_plan',
            'general': 'answer',
            'ui_designer': 'design',
            'qa_tester': 'test',
            'security_auditor': 'analyze',
            'database_expert': 'design',
            'devops': 'deploy',
            'api': 'design',
            'flutter': 'generate',
            'react': 'generate'
        }
        
        # Check for review/optimization keywords
        user_input_lower = user_input.lower()
        if 'review' in user_input_lower or 'optimize' in user_input_lower or 'best practices' in user_input_lower:
            if agent_name == 'code':
                return 'review'
        
        return task_type_map.get(agent_name, 'general')
    
    def _synthesize_final_product(self, user_input: str, execution_results: List[Dict], context: Dict) -> Dict:
        """Synthesize a cohesive final product from all agent results."""
        # Build a summary of what was accomplished
        synthesis_prompt = f"""
You are synthesizing the final output from multiple specialized agents.

Original Request: {user_input}

Agent Results:
"""
        
        for result in execution_results:
            agent_name = result.get('agent', 'unknown')
            status = result.get('status', 'unknown')
            task = result.get('task', '')
            
            synthesis_prompt += f"\n- {agent_name} ({status}): {task}\n"
            
            if status == 'success' and 'result' in result:
                agent_result = result['result']
                if 'design_spec' in agent_result:
                    synthesis_prompt += f"  Design: {agent_result['design_spec'][:500]}...\n"
                if 'code' in agent_result:
                    synthesis_prompt += f"  Code provided\n"
                if 'review' in agent_result:
                    synthesis_prompt += f"  Review provided\n"
        
        synthesis_prompt += """
Synthesize this into a cohesive final product that includes:
1. A summary of what was created
2. The complete code (if applicable)
3. Design specifications (if applicable)
4. Any reviews or optimizations
5. Clear instructions on how to use the final product

Provide the final product in a well-structured, ready-to-use format.
"""
        
        messages = [
            {"role": "system", "content": "You are a synthesis agent that combines work from multiple specialized agents into a cohesive final product."},
            {"role": "user", "content": synthesis_prompt}
        ]
        
        try:
            final_synthesis = call_llm(messages)
            
            return {
                'summary': 'Multi-agent collaboration completed successfully',
                'synthesis': final_synthesis,
                'agents_used': [r['agent'] for r in execution_results],
                'all_results': context
            }
        except Exception as e:
            # If synthesis fails, return the raw results
            return {
                'summary': 'Multi-agent collaboration completed (synthesis failed)',
                'error': str(e),
                'agents_used': [r['agent'] for r in execution_results],
                'all_results': context
            }
    
    def route_task(self, task_type: str, task_data: Dict) -> Dict:
        """Route a task to the appropriate agent based on type with RAG context."""
        agent = self.agents.get(task_type)
        if not agent:
            raise ValueError(f"Unknown agent type: {task_type}")
        
        # RAG Workflow: Retrieve relevant context from memory before executing
        user_input = task_data.get('user_input', task_data.get('task', task_data.get('question', '')))
        if user_input:
            relevant_context = self._retrieve_relevant_context(user_input, task_type)
            if relevant_context:
                # Add retrieved context to task_data
                task_data['rag_context'] = relevant_context
        
        result = agent.execute(task_data)
        
        # Save conversation to memory after execution
        if user_input and 'result' in result:
            self._save_conversation_to_memory(user_input, result, task_type, task_data)
        
        return result
    
    def _retrieve_relevant_context(self, query: str, task_type: str) -> Dict:
        """Retrieve relevant context from memory based on query and task type."""
        try:
            # Determine which collections to search based on task type
            collections_to_search = []
            
            if task_type == 'code':
                collections_to_search = ['code', 'errors', 'projects']
            elif task_type == 'docs':
                collections_to_search = ['documents', 'notes']
            elif task_type in ['react', 'flutter', 'api']:
                collections_to_search = ['code', 'documents', 'projects']
            elif task_type == 'general':
                collections_to_search = ['documents', 'notes', 'conversation']
            else:
                collections_to_search = ['documents', 'code', 'notes']
            
            all_results = {}
            
            for collection in collections_to_search:
                search_result = self.memory_agent.search_memory(
                    query=query,
                    collection=collection,
                    n_results=3
                )
                
                if search_result.get('status') == 'searched' and search_result.get('results'):
                    all_results[collection] = {
                        'results': search_result['results'],
                        'metadatas': search_result['metadatas'],
                        'distances': search_result['distances']
                    }
            
            return all_results if all_results else None
            
        except Exception as e:
            # If retrieval fails, continue without context
            print(f"Error retrieving context from memory: {e}")
            return None
    
    def _save_conversation_to_memory(self, user_input: str, result: Dict, 
                                    agent_type: str, task_data: Dict) -> None:
        """Save the conversation turn to memory with rich metadata."""
        try:
            # Extract the response from the result
            if isinstance(result, dict):
                response = str(result.get('result', result.get('answer', str(result))))
            else:
                response = str(result)
            
            # Get project from task_data or context
            project = task_data.get('project') or task_data.get('context', {}).get('project')
            
            # Save conversation
            self.memory_agent.save_conversation(
                user_message=user_input,
                assistant_message=response,
                project=project,
                agent=agent_type
            )
            
        except Exception as e:
            # If saving fails, log but don't interrupt the flow
            print(f"Error saving conversation to memory: {e}")
    
    def execute_plan(self, plan: List[Dict]) -> List[Dict]:
        """Execute a multi-step plan coordinated by the planner."""
        results = []
        for step in plan:
            agent_name = step.get('agent')
            task_data = step.get('task', {})
            result = self.route_task(agent_name, task_data)
            results.append(result)
        return results
    
    def get_agent_status(self) -> Dict[str, str]:
        """Get status of all agents."""
        return {
            name: agent.get_status()
            for name, agent in self.agents.items()
        }
    
    # New Architecture Methods
    
    async def execute_with_new_architecture(self, user_input: str) -> Dict:
        """Execute task using the new planner-executor architecture."""
        # Log task start
        self.logger.info("Manager", f"Starting task with new architecture: {user_input}")
        
        # Register agents to registry (if not already registered)
        if self.agent_registry.get_count() == 0:
            self.register_agents_to_registry()
        
        # Initialize shared state
        self.shared_state.initialize_task(user_input)
        
        # Create plan using planner
        task_graph = self.planner.create_plan(user_input, context={"project": "current"})
        
        self.logger.info("Planner", f"Created plan with {len(task_graph.steps)} steps")
        
        # Update executor with agent registry
        self.executor.agent_registry = self.agent_registry
        
        # Execute plan using executor (already async context)
        execution_result = await self.executor.execute_plan(task_graph)
        
        # Log completion
        self.logger.info("Executor", f"Plan execution completed: {execution_result['success']}")
        
        return {
            "goal": user_input,
            "plan": task_graph.to_dict(),
            "execution": execution_result,
            "shared_state": self.shared_state.get_all()
        }
    
    def setup_event_handlers(self):
        """Set up event handlers for logging."""
        async def log_plan_started(event):
            self.logger.info("EventBus", f"Plan started: {event.data.get('goal')}")
        
        async def log_step_completed(event):
            self.logger.info("EventBus", f"Step {event.data.get('step')} completed by {event.data.get('agent')}")
        
        async def log_plan_failed(event):
            self.logger.error("EventBus", f"Plan failed at step {event.data.get('failed_step')}: {event.data.get('error')}")
        
        async def log_plan_completed(event):
            self.logger.info("EventBus", f"Plan completed: {event.data.get('successful_steps')}/{event.data.get('total_steps')} steps successful")
        
        # Register handlers
        self.event_bus.subscribe("plan_started", log_plan_started)
        self.event_bus.subscribe("step_completed", log_step_completed)
        self.event_bus.subscribe("plan_failed", log_plan_failed)
        self.event_bus.subscribe("plan_completed", log_plan_completed)
    
    def register_agents_to_registry(self):
        """Register existing agents to the new agent registry."""
        from registry.agent_adapter import AgentAdapter
        
        # Register all existing agents using the adapter
        self.agent_registry.register(
            AgentAdapter("code", self.code_agent),
            metadata={"type": "code", "description": "Code analysis and generation"}
        )
        self.agent_registry.register(
            AgentAdapter("memory", self.memory_agent),
            metadata={"type": "memory", "description": "Memory management"}
        )
        self.agent_registry.register(
            AgentAdapter("docs", self.docs_agent),
            metadata={"type": "docs", "description": "Documentation handling"}
        )
        self.agent_registry.register(
            AgentAdapter("planner", self.planner_agent),
            metadata={"type": "planner", "description": "Task planning"}
        )
        self.agent_registry.register(
            AgentAdapter("general", self.general_agent),
            metadata={"type": "general", "description": "General tasks"}
        )
        self.agent_registry.register(
            AgentAdapter("ui_designer", self.ui_designer_agent),
            metadata={"type": "ui", "description": "UI/UX design"}
        )
        self.agent_registry.register(
            AgentAdapter("qa_tester", self.qa_tester_agent),
            metadata={"type": "qa", "description": "Quality assurance"}
        )
        self.agent_registry.register(
            AgentAdapter("security_auditor", self.security_auditor_agent),
            metadata={"type": "security", "description": "Security auditing"}
        )
        self.agent_registry.register(
            AgentAdapter("database_expert", self.database_expert_agent),
            metadata={"type": "database", "description": "Database expertise"}
        )
        self.agent_registry.register(
            AgentAdapter("devops", self.devops_agent),
            metadata={"type": "devops", "description": "DevOps operations"}
        )
        self.agent_registry.register(
            AgentAdapter("api", self.api_agent),
            metadata={"type": "api", "description": "API development"}
        )
        self.agent_registry.register(
            AgentAdapter("flutter", self.flutter_agent),
            metadata={"type": "flutter", "description": "Flutter development"}
        )
        self.agent_registry.register(
            AgentAdapter("react", self.react_agent),
            metadata={"type": "react", "description": "React development"}
        )
        
        self.logger.info("Manager", f"Registered {self.agent_registry.get_count()} agents to registry")


# Keep backward compatibility
AgentManager = ManagerAgent
