from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .base_tool import BaseTool


class ToolManager:
    """Central manager for all tools with registration, permissions, and logging."""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.tool_permissions: Dict[str, List[str]] = {}
        self.usage_log: List[Dict] = []
        self.logger = logging.getLogger(__name__)
    
    def register_tool(self, tool: BaseTool, permissions: Optional[List[str]] = None) -> None:
        """Register a tool with optional permissions.
        
        Args:
            tool: Tool instance to register
            permissions: Optional list of required permissions
        """
        if not isinstance(tool, BaseTool):
            raise TypeError(f"Tool must inherit from BaseTool, got {type(tool)}")
        
        tool_name = tool.name
        self.tools[tool_name] = tool
        self.tool_permissions[tool_name] = permissions or tool.permissions
        
        self.logger.info(f"Registered tool: {tool_name} with permissions: {self.tool_permissions[tool_name]}")
    
    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool.
        
        Args:
            tool_name: Name of tool to unregister
            
        Returns:
            bool: True if unregistered, False if not found
        """
        if tool_name in self.tools:
            del self.tools[tool_name]
            del self.tool_permissions[tool_name]
            self.logger.info(f"Unregistered tool: {tool_name}")
            return True
        return False
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name.
        
        Args:
            tool_name: Name of tool to retrieve
            
        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools with their metadata.
        
        Returns:
            List of tool information dictionaries
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "permissions": self.tool_permissions[tool.name],
                "requires_auth": tool.requires_auth
            }
            for tool in self.tools.values()
        ]
    
    def check_permissions(self, tool_name: str, user_permissions: List[str]) -> bool:
        """Check if user has required permissions for a tool.
        
        Args:
            tool_name: Name of tool to check
            user_permissions: List of user's permissions
            
        Returns:
            bool: True if user has required permissions
        """
        required_permissions = self.tool_permissions.get(tool_name, [])
        
        # If no permissions required, allow access
        if not required_permissions:
            return True
        
        # Check if user has all required permissions
        return all(perm in user_permissions for perm in required_permissions)
    
    def execute_tool(self, tool_name: str, user_permissions: Optional[List[str]] = None, 
                     log_usage: bool = True, **kwargs) -> Dict[str, Any]:
        """Execute a tool with permission checking and logging.
        
        Args:
            tool_name: Name of tool to execute
            user_permissions: Optional list of user permissions
            log_usage: Whether to log this execution
            **kwargs: Tool-specific parameters
            
        Returns:
            Dict with execution result
        """
        tool = self.get_tool(tool_name)
        
        if not tool:
            return {
                "success": False,
                "error": f"Tool not found: {tool_name}",
                "available_tools": list(self.tools.keys())
            }
        
        # Check permissions if provided
        if user_permissions is not None and not self.check_permissions(tool_name, user_permissions):
            return {
                "success": False,
                "error": f"Permission denied for tool: {tool_name}",
                "required_permissions": self.tool_permissions[tool_name]
            }
        
        # Validate parameters
        if not tool.validate_params(**kwargs):
            return {
                "success": False,
                "error": f"Parameter validation failed for tool: {tool_name}"
            }
        
        # Log usage if requested
        if log_usage:
            self._log_execution(tool_name, kwargs)
        
        # Execute tool
        try:
            result = tool.execute(**kwargs)
            
            # Add success flag if not present
            if "success" not in result:
                result["success"] = "error" not in result
            
            return result
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name
            }
    
    def _log_execution(self, tool_name: str, params: Dict) -> None:
        """Log tool execution for auditing.
        
        Args:
            tool_name: Name of executed tool
            params: Execution parameters
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "params": params
        }
        self.usage_log.append(log_entry)
        
        # Keep log size manageable (last 1000 entries)
        if len(self.usage_log) > 1000:
            self.usage_log = self.usage_log[-1000:]
    
    def get_usage_log(self, tool_name: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get usage log, optionally filtered by tool.
        
        Args:
            tool_name: Optional tool name to filter by
            limit: Maximum number of entries to return
            
        Returns:
            List of log entries
        """
        if tool_name:
            filtered = [entry for entry in self.usage_log if entry["tool"] == tool_name]
        else:
            filtered = self.usage_log
        
        return filtered[-limit:]
    
    def clear_usage_log(self) -> None:
        """Clear the usage log."""
        self.usage_log = []
        self.logger.info("Usage log cleared")
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get statistics about tool usage.
        
        Returns:
            Dict with usage statistics
        """
        tool_counts = {}
        for entry in self.usage_log:
            tool_name = entry["tool"]
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        
        return {
            "total_tools": len(self.tools),
            "total_executions": len(self.usage_log),
            "tool_counts": tool_counts,
            "most_used_tool": max(tool_counts.items(), key=lambda x: x[1])[0] if tool_counts else None
        }
