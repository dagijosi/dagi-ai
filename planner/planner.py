from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class TaskStep:
    """A single step in the task graph."""
    step: int
    agent: str
    action: str
    description: str
    dependencies: List[int] = field(default_factory=list)
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    expected_output: Optional[str] = None


@dataclass
class TaskGraph:
    """A complete task graph representing a plan."""
    goal: str
    steps: List[TaskStep]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert task graph to dictionary."""
        return {
            "goal": self.goal,
            "created_at": self.created_at,
            "steps": [
                {
                    "step": step.step,
                    "agent": step.agent,
                    "action": step.action,
                    "description": step.description,
                    "dependencies": step.dependencies,
                    "context_requirements": step.context_requirements,
                    "expected_output": step.expected_output
                }
                for step in self.steps
            ],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TaskGraph':
        """Create task graph from dictionary."""
        steps = [
            TaskStep(
                step=step_data["step"],
                agent=step_data["agent"],
                action=step_data["action"],
                description=step_data["description"],
                dependencies=step_data.get("dependencies", []),
                context_requirements=step_data.get("context_requirements", {}),
                expected_output=step_data.get("expected_output")
            )
            for step_data in data["steps"]
        ]
        return cls(
            goal=data["goal"],
            steps=steps,
            created_at=data.get("created_at", datetime.now().isoformat()),
            metadata=data.get("metadata", {})
        )


class Planner:
    """Planning engine that generates task graphs from user requests."""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.agent_capabilities = {
            "code": ["read_code", "write_code", "analyze_code", "debug_code"],
            "docs": ["read_docs", "search_docs", "answer_questions"],
            "search": ["find_files", "search_content", "semantic_search"],
            "memory": ["save_memory", "retrieve_memory", "update_memory"],
            "terminal": ["execute_command", "run_tests"],
            "git": ["status", "commit", "push", "pull", "diff", "history"],
            "ui": ["design_ui", "generate_components"],
            "api": ["design_api", "implement_endpoints"],
            "flutter": ["create_flutter_components", "mobile_ui"],
            "react": ["create_react_components", "web_ui"],
            "qa": ["write_tests", "run_tests", "analyze_coverage"],
            "security": ["audit_code", "check_vulnerabilities"],
            "database": ["design_schema", "write_queries", "optimize"],
            "devops": ["deploy", "configure_ci", "monitor"]
        }
    
    def create_plan(self, goal: str, context: Optional[Dict] = None) -> TaskGraph:
        """Create a task graph for the given goal."""
        context = context or {}
        
        # Analyze the goal to determine required steps
        steps = self._analyze_goal(goal, context)
        
        return TaskGraph(
            goal=goal,
            steps=steps,
            metadata={
                "context": context,
                "estimated_steps": len(steps)
            }
        )
    
    def _analyze_goal(self, goal: str, context: Dict) -> List[TaskStep]:
        """Analyze the goal and generate task steps."""
        # This is an enhanced implementation that utilizes all registered agents
        steps = []
        step_counter = 1
        goal_lower = goal.lower()
        
        # Step 1: Always start with understanding the current state
        if self._needs_code_analysis(goal):
            # Extract relevant terms for search (avoid common words)
            search_terms = [word for word in goal.split() if word.lower() not in ["write", "create", "build", "make", "a", "an", "the", "with", "and"]]
            search_filename = search_terms[0] if search_terms else "*"
            
            steps.append(TaskStep(
                step=step_counter,
                agent="search",
                action="filename",
                description="Find relevant files for the task",
                context_requirements={"operation": "filename", "filename": search_filename, "search_path": "."}
            ))
            step_counter += 1
        
        # Step 2: UI Design for UI-related tasks
        if self._needs_ui_design(goal):
            steps.append(TaskStep(
                step=step_counter,
                agent="ui_designer",
                action="design",
                description=f"Design UI/UX for: {goal}",
                dependencies=[step_counter - 1] if step_counter > 1 else [],
                context_requirements={"type": "design", "task": goal, "goal": goal}
            ))
            step_counter += 1
        
        # Step 3: Framework-specific implementation
        framework_agent = self._detect_framework(goal)
        if framework_agent:
            steps.append(TaskStep(
                step=step_counter,
                agent=framework_agent,
                action="generate",
                description=f"Generate {framework_agent.upper()} implementation for: {goal}",
                dependencies=[step_counter - 1] if step_counter > 1 else [],
                context_requirements={"type": "generate", "task": goal, "goal": goal, "framework": framework_agent}
            ))
            step_counter += 1
        
        # Step 4: Read existing code if needed (for modifications)
        if self._needs_code_reading(goal):
            steps.append(TaskStep(
                step=step_counter,
                agent="file",
                action="read",
                description="Read and analyze existing code",
                dependencies=[step_counter - 1] if step_counter > 1 else [],
                context_requirements={"file_path": "main.py"}
            ))
            step_counter += 1
        
        # Step 5: Generate new code/changes
        if self._needs_code_generation(goal):
            steps.append(TaskStep(
                step=step_counter,
                agent="code",
                action="generate",
                description=f"Generate code for: {goal}",
                dependencies=[step_counter - 1] if step_counter > 1 else [],
                context_requirements={"type": "generate", "task": goal, "goal": goal, "use_design_spec": True}
            ))
            step_counter += 1
        
        # Step 6: API design for API-related tasks
        if self._needs_api_design(goal):
            steps.append(TaskStep(
                step=step_counter,
                agent="api",
                action="design",
                description="Design API endpoints and structure",
                dependencies=[step_counter - 1] if step_counter > 1 else [],
                context_requirements={"type": "design", "task": goal}
            ))
            step_counter += 1
        
        # Step 7: Database design for database-related tasks
        if self._needs_database_design(goal):
            steps.append(TaskStep(
                step=step_counter,
                agent="database_expert",
                action="design",
                description="Design database schema and queries",
                dependencies=[step_counter - 1] if step_counter > 1 else [],
                context_requirements={"type": "design", "task": goal}
            ))
            step_counter += 1
        
        # Step 8: Security audit for security-related tasks
        if self._needs_security_audit(goal):
            steps.append(TaskStep(
                step=step_counter,
                agent="security_auditor",
                action="analyze",
                description="Perform security analysis and audit",
                dependencies=[step_counter - 1] if step_counter > 1 else [],
                context_requirements={"type": "analyze", "task": goal}
            ))
            step_counter += 1
        
        # Step 9: Write files
        if self._needs_file_writing(goal):
            steps.append(TaskStep(
                step=step_counter,
                agent="file",
                action="write_files",
                description="Write generated code to files",
                dependencies=[step_counter - 1] if step_counter > 1 else [],
                context_requirements={"operation": "write", "code_from_previous_step": True}
            ))
            step_counter += 1
        
        # Step 10: Run tests if applicable
        if self._needs_testing(goal):
            steps.append(TaskStep(
                step=step_counter,
                agent="qa_tester",
                action="test",
                description="Write and run tests to verify changes",
                dependencies=[step_counter - 1] if step_counter > 1 else [],
                context_requirements={"type": "test", "task": goal}
            ))
            step_counter += 1
        
        # Step 11: DevOps operations if applicable
        if self._needs_devops(goal):
            steps.append(TaskStep(
                step=step_counter,
                agent="devops",
                action="deploy",
                description="Handle deployment and infrastructure",
                dependencies=[step_counter - 1] if step_counter > 1 else [],
                context_requirements={"type": "deploy", "task": goal}
            ))
            step_counter += 1
        
        # Step 12: Commit changes if applicable
        if self._needs_git_commit(goal):
            steps.append(TaskStep(
                step=step_counter,
                agent="git",
                action="commit",
                description="Commit changes to git",
                dependencies=[step_counter - 1] if step_counter > 1 else [],
                context_requirements={"commit_message": f"Implement: {goal}"}
            ))
            step_counter += 1
        
        # Step 13: Save to memory (always)
        steps.append(TaskStep(
            step=step_counter,
            agent="memory",
            action="save_context",
            description="Save task context to memory for future reference",
            dependencies=[step_counter - 1] if step_counter > 1 else [],
            context_requirements={"operation": "save_conversation", "goal": goal, "results": "all_previous_steps", "document": f"Task: {goal}\nCompleted successfully with multi-agent coordination."}
        ))
        
        return steps
    
    def _needs_code_analysis(self, goal: str) -> bool:
        """Determine if code analysis is needed."""
        # Most tasks need code analysis to understand current state
        keywords = ["create", "update", "modify", "fix", "implement", "add", "build", "design", "write", "generate", "make"]
        return any(keyword in goal.lower() for keyword in keywords)
    
    def _needs_code_reading(self, goal: str) -> bool:
        """Determine if code reading is needed."""
        keywords = ["update", "modify", "fix", "refactor", "improve"]
        return any(keyword in goal.lower() for keyword in keywords)
    
    def _needs_code_generation(self, goal: str) -> bool:
        """Determine if code generation is needed."""
        keywords = ["create", "implement", "add", "build", "generate", "write", "make"]
        return any(keyword in goal.lower() for keyword in keywords)
    
    def _needs_file_writing(self, goal: str) -> bool:
        """Determine if file writing is needed."""
        return self._needs_code_generation(goal)
    
    def _needs_testing(self, goal: str) -> bool:
        """Determine if testing is needed."""
        keywords = ["test", "verify", "validate"]
        return any(keyword in goal.lower() for keyword in keywords)
    
    def _needs_git_commit(self, goal: str) -> bool:
        """Determine if git commit is needed."""
        keywords = ["commit", "save", "push"]
        return any(keyword in goal.lower() for keyword in keywords)
    
    def _needs_ui_design(self, goal: str) -> bool:
        """Determine if UI design is needed."""
        # More specific UI keywords to avoid triggering for non-UI tasks
        ui_keywords = ["ui", "interface", "layout", "component", "page", "screen", "frontend", "web page", "login page", "dashboard"]
        # Exclude these keywords that might trigger UI but aren't really UI tasks
        exclude_keywords = ["api", "database", "schema", "security", "backend", "service"]
        
        goal_lower = goal.lower()
        
        # Check if it's a UI task
        has_ui_keyword = any(keyword in goal_lower for keyword in ui_keywords)
        # Check if it should be excluded
        has_exclude_keyword = any(keyword in goal_lower for keyword in exclude_keywords)
        
        return has_ui_keyword and not has_exclude_keyword
    
    def _detect_framework(self, goal: str) -> Optional[str]:
        """Detect which framework is needed for the task."""
        goal_lower = goal.lower()
        if "react" in goal_lower:
            return "react"
        elif "flutter" in goal_lower:
            return "flutter"
        return None
    
    def _needs_api_design(self, goal: str) -> bool:
        """Determine if API design is needed."""
        keywords = ["api", "endpoint", "rest", "graphql", "service", "backend"]
        return any(keyword in goal.lower() for keyword in keywords)
    
    def _needs_database_design(self, goal: str) -> bool:
        """Determine if database design is needed."""
        keywords = ["database", "schema", "sql", "query", "migration", "data"]
        return any(keyword in goal.lower() for keyword in keywords)
    
    def _needs_security_audit(self, goal: str) -> bool:
        """Determine if security audit is needed."""
        keywords = ["security", "audit", "vulnerability", "auth", "authentication", "authorization"]
        return any(keyword in goal.lower() for keyword in keywords)
    
    def _needs_devops(self, goal: str) -> bool:
        """Determine if DevOps operations are needed."""
        keywords = ["deploy", "ci/cd", "pipeline", "infrastructure", "docker", "kubernetes"]
        return any(keyword in goal.lower() for keyword in keywords)
    
    def get_suitable_agents(self, action: str) -> List[str]:
        """Get agents that can perform a given action."""
        suitable = []
        for agent, capabilities in self.agent_capabilities.items():
            if action in capabilities:
                suitable.append(agent)
        return suitable
    
    def register_agent_capability(self, agent: str, capabilities: List[str]):
        """Register capabilities for an agent."""
        if agent not in self.agent_capabilities:
            self.agent_capabilities[agent] = []
        self.agent_capabilities[agent].extend(capabilities)
