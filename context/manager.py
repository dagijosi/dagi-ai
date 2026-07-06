from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ContextRequest:
    """Request for context building."""
    task: str
    agent: str
    step: int
    requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextBundle:
    """Bundle of context for a task."""
    task: str
    agent: str
    step: int
    memory_context: List[Dict] = field(default_factory=list)
    file_context: Dict[str, str] = field(default_factory=dict)
    conversation_context: List[Dict] = field(default_factory=list)
    documentation_context: List[Dict] = field(default_factory=list)
    project_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_total_size(self) -> int:
        """Estimate total context size in characters."""
        size = 0
        size += sum(len(str(m)) for m in self.memory_context)
        size += sum(len(content) for content in self.file_context.values())
        size += sum(len(str(c)) for c in self.conversation_context)
        size += sum(len(str(d)) for d in self.documentation_context)
        size += len(str(self.project_context))
        return size


class ContextManager:
    """Manages context building for agents and tools."""
    
    def __init__(self, memory_agent=None, file_tool=None, search_tool=None):
        self.memory_agent = memory_agent
        self.file_tool = file_tool
        self.search_tool = search_tool
        self.max_context_size = 100000  # Max characters in context
        self.project_root = Path(".")
    
    def build_context(self, request: ContextRequest) -> ContextBundle:
        """Build context bundle for a task."""
        bundle = ContextBundle(
            task=request.task,
            agent=request.agent,
            step=request.step
        )
        
        # Build different types of context based on requirements
        if request.requirements.get("memory", True):
            bundle.memory_context = self._build_memory_context(request)
        
        if request.requirements.get("files", True):
            bundle.file_context = self._build_file_context(request)
        
        if request.requirements.get("conversation", False):
            bundle.conversation_context = self._build_conversation_context(request)
        
        if request.requirements.get("documentation", False):
            bundle.documentation_context = self._build_documentation_context(request)
        
        if request.requirements.get("project", True):
            bundle.project_context = self._build_project_context(request)
        
        # Trim context if too large
        bundle = self._trim_context(bundle)
        
        return bundle
    
    def _build_memory_context(self, request: ContextRequest) -> List[Dict]:
        """Build memory context from memory agent."""
        if not self.memory_agent:
            return []
        
        try:
            # Search memory for relevant context
            results = self.memory_agent.search_memory(
                query=request.task,
                collection="documents",
                n_results=5
            )
            
            return results if isinstance(results, list) else []
        except Exception as e:
            print(f"Error building memory context: {e}")
            return []
    
    def _build_file_context(self, request: ContextRequest) -> Dict[str, str]:
        """Build file context based on task requirements."""
        if not self.file_tool or not self.search_tool:
            return {}
        
        file_context = {}
        
        # Search for relevant files
        try:
            # First, find files related to the task
            search_result = self.search_tool.execute(
                operation="filename",
                filename=request.task.split()[0] if request.task else "*",
                search_path="."
            )
            
            if search_result.get("status") == "found":
                files = search_result.get("matches", [])[:5]  # Limit to 5 files
                
                # Read the files
                for file_path in files:
                    try:
                        read_result = self.file_tool.execute(
                            operation="read",
                            file_path=file_path
                        )
                        
                        if read_result.get("status") == "read":
                            file_context[file_path] = read_result.get("content", "")
                    except Exception:
                        continue
        except Exception as e:
            print(f"Error building file context: {e}")
        
        return file_context
    
    def _build_conversation_context(self, request: ContextRequest) -> List[Dict]:
        """Build conversation context from memory."""
        if not self.memory_agent:
            return []
        
        try:
            # Get recent conversation history
            results = self.memory_agent.search_memory(
                query=request.task,
                collection="conversation",
                n_results=3
            )
            
            return results if isinstance(results, list) else []
        except Exception as e:
            print(f"Error building conversation context: {e}")
            return []
    
    def _build_documentation_context(self, request: ContextRequest) -> List[Dict]:
        """Build documentation context from memory."""
        if not self.memory_agent:
            return []
        
        try:
            # Search documentation
            results = self.memory_agent.search_memory(
                query=request.task,
                collection="notes",
                n_results=3
            )
            
            return results if isinstance(results, list) else []
        except Exception as e:
            print(f"Error building documentation context: {e}")
            return []
    
    def _build_project_context(self, request: ContextRequest) -> Dict[str, Any]:
        """Build project context."""
        project_context = {
            "project_root": str(self.project_root),
            "structure": self._get_project_structure(),
            "tech_stack": self._detect_tech_stack()
        }
        
        return project_context
    
    def _get_project_structure(self) -> Dict[str, Any]:
        """Get project directory structure."""
        structure = {}
        
        try:
            for item in self.project_root.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    structure[item.name] = "directory"
                elif item.is_file() and not item.name.startswith('.'):
                    structure[item.name] = "file"
        except Exception as e:
            print(f"Error getting project structure: {e}")
        
        return structure
    
    def _detect_tech_stack(self) -> List[str]:
        """Detect technology stack from project files."""
        tech_stack = []
        
        try:
            # Check for common tech indicators
            if (self.project_root / "package.json").exists():
                tech_stack.append("Node.js")
            
            if (self.project_root / "requirements.txt").exists():
                tech_stack.append("Python")
            
            if (self.project_root / "pom.xml").exists():
                tech_stack.append("Java/Maven")
            
            if (self.project_root / "go.mod").exists():
                tech_stack.append("Go")
            
            if (self.project_root / "Cargo.toml").exists():
                tech_stack.append("Rust")
            
            if (self.project_root / "pubspec.yaml").exists():
                tech_stack.append("Flutter/Dart")
        except Exception as e:
            print(f"Error detecting tech stack: {e}")
        
        return tech_stack
    
    def _trim_context(self, bundle: ContextBundle) -> ContextBundle:
        """Trim context to fit within size limits."""
        current_size = bundle.get_total_size()
        
        if current_size <= self.max_context_size:
            return bundle
        
        # Trim file context first (largest usually)
        while bundle.get_total_size() > self.max_context_size and bundle.file_context:
            # Remove the largest file
            largest_file = max(bundle.file_context.items(), key=lambda x: len(x[1]))
            del bundle.file_context[largest_file[0]]
        
        # Trim memory context
        while bundle.get_total_size() > self.max_context_size and bundle.memory_context:
            bundle.memory_context.pop()
        
        # Trim conversation context
        while bundle.get_total_size() > self.max_context_size and bundle.conversation_context:
            bundle.conversation_context.pop()
        
        # Trim documentation context
        while bundle.get_total_size() > self.max_context_size and bundle.documentation_context:
            bundle.documentation_context.pop()
        
        return bundle
    
    def set_max_context_size(self, size: int):
        """Set maximum context size."""
        self.max_context_size = size
    
    def set_project_root(self, path: str):
        """Set project root directory."""
        self.project_root = Path(path)
