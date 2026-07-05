from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseTool(ABC):
    """Base class for all tools in the system."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name identifier."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for documentation and discovery."""
        pass
    
    @property
    def permissions(self) -> list:
        """Required permissions for this tool."""
        return []
    
    @property
    def requires_auth(self) -> bool:
        """Whether this tool requires authentication."""
        return False
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            Dict with execution result
        """
        pass
    
    def validate_params(self, **kwargs) -> bool:
        """Validate parameters before execution.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        return True
    
    def log_usage(self, **kwargs):
        """Log tool usage for auditing and monitoring.
        
        Args:
            **kwargs: Execution parameters for logging
        """
        # Default implementation - can be overridden
        pass
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"
