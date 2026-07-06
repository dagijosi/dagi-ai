from typing import Dict, List, Optional, Any, Type
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Base class for all agents."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent with given context."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities."""
        pass


class AgentRegistry:
    """Registry for managing and discovering agents."""
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_capabilities: Dict[str, List[str]] = {}
        self._agent_metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(self, agent: BaseAgent, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register an agent."""
        self._agents[agent.name] = agent
        self._agent_capabilities[agent.name] = agent.get_capabilities()
        self._agent_metadata[agent.name] = metadata or {}
    
    def unregister(self, agent_name: str) -> None:
        """Unregister an agent."""
        if agent_name in self._agents:
            del self._agents[agent_name]
            del self._agent_capabilities[agent_name]
            del self._agent_metadata[agent_name]
    
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self._agents.get(agent_name)
    
    def get_all_agents(self) -> Dict[str, BaseAgent]:
        """Get all registered agents."""
        return self._agents.copy()
    
    def get_agent_names(self) -> List[str]:
        """Get list of all agent names."""
        return list(self._agents.keys())
    
    def get_capabilities(self, agent_name: str) -> List[str]:
        """Get capabilities of a specific agent."""
        return self._agent_capabilities.get(agent_name, [])
    
    def get_all_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all agents."""
        return self._agent_capabilities.copy()
    
    def find_agents_by_capability(self, capability: str) -> List[str]:
        """Find agents that have a specific capability."""
        return [
            agent_name
            for agent_name, capabilities in self._agent_capabilities.items()
            if capability in capabilities
        ]
    
    def find_suitable_agents(self, task: str) -> List[str]:
        """Find agents suitable for a task based on task description."""
        suitable_agents = []
        task_lower = task.lower()
        
        for agent_name, capabilities in self._agent_capabilities.items():
            # Check if any capability matches the task
            for capability in capabilities:
                if capability.lower() in task_lower or task_lower in capability.lower():
                    suitable_agents.append(agent_name)
                    break
        
        return suitable_agents
    
    def get_metadata(self, agent_name: str) -> Dict[str, Any]:
        """Get metadata for an agent."""
        return self._agent_metadata.get(agent_name, {})
    
    def set_metadata(self, agent_name: str, metadata: Dict[str, Any]) -> None:
        """Set metadata for an agent."""
        if agent_name in self._agents:
            self._agent_metadata[agent_name].update(metadata)
    
    def is_registered(self, agent_name: str) -> bool:
        """Check if an agent is registered."""
        return agent_name in self._agents
    
    def get_count(self) -> int:
        """Get total number of registered agents."""
        return len(self._agents)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents with their information."""
        return [
            {
                "name": agent_name,
                "capabilities": self._agent_capabilities.get(agent_name, []),
                "metadata": self._agent_metadata.get(agent_name, {})
            }
            for agent_name in self._agents.keys()
        ]
