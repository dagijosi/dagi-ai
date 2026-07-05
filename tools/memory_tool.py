from typing import Dict, List, Optional, Any

from .base_tool import BaseTool


class MemoryTool(BaseTool):
    """Tool for memory operations using the Memory Agent."""
    
    def __init__(self, memory_agent=None):
        self.memory_agent = memory_agent
    
    @property
    def name(self) -> str:
        return "memory"
    
    @property
    def description(self) -> str:
        return "Memory operations: save, search, delete, update memories and conversations"
    
    @property
    def permissions(self) -> list:
        return ["memory_read", "memory_write"]
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute a memory operation."""
        operation = kwargs.get('operation')
        
        if not self.memory_agent:
            return {
                "success": False,
                "error": "Memory agent not initialized"
            }
        
        try:
            if operation == 'save':
                result = self._save_memory(kwargs)
            elif operation == 'search':
                result = self._search_memory(kwargs)
            elif operation == 'delete':
                result = self._delete_memory(kwargs)
            elif operation == 'update':
                result = self._update_memory(kwargs)
            elif operation == 'save_conversation':
                result = self._save_conversation(kwargs)
            elif operation == 'get_collections':
                result = self._get_collections()
            elif operation == 'get_stats':
                result = self._get_stats()
            else:
                result = {
                    "success": False,
                    "error": f"Unknown operation: {operation}"
                }
            
            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _save_memory(self, kwargs: Dict) -> Dict[str, Any]:
        """Save a memory with rich metadata."""
        content = kwargs.get('content')
        collection = kwargs.get('collection', 'documents')
        metadata = kwargs.get('metadata', {})
        
        result = self.memory_agent.save_memory(content, collection, metadata)
        
        return {
            "success": "error" not in result,
            **result
        }
    
    def _search_memory(self, kwargs: Dict) -> Dict[str, Any]:
        """Search for relevant memories."""
        query = kwargs.get('query')
        collection = kwargs.get('collection', 'documents')
        n_results = kwargs.get('n_results', 5)
        where = kwargs.get('where', None)
        
        result = self.memory_agent.search_memory(query, collection, n_results, where)
        
        return {
            "success": "error" not in result,
            **result
        }
    
    def _delete_memory(self, kwargs: Dict) -> Dict[str, Any]:
        """Delete a memory by ID."""
        memory_id = kwargs.get('memory_id')
        collection = kwargs.get('collection', 'documents')
        
        result = self.memory_agent.delete_memory(memory_id, collection)
        
        return {
            "success": "error" not in result,
            **result
        }
    
    def _update_memory(self, kwargs: Dict) -> Dict[str, Any]:
        """Update a memory's content and/or metadata."""
        memory_id = kwargs.get('memory_id')
        collection = kwargs.get('collection', 'documents')
        content = kwargs.get('content', None)
        metadata = kwargs.get('metadata', None)
        
        result = self.memory_agent.update_memory(memory_id, collection, content, metadata)
        
        return {
            "success": "error" not in result,
            **result
        }
    
    def _save_conversation(self, kwargs: Dict) -> Dict[str, Any]:
        """Save a conversation turn."""
        user_message = kwargs.get('user_message')
        assistant_message = kwargs.get('assistant_message')
        project = kwargs.get('project', None)
        agent = kwargs.get('agent', None)
        conversation_id = kwargs.get('conversation_id', None)
        
        result = self.memory_agent.save_conversation(
            user_message, assistant_message, project, agent, conversation_id
        )
        
        return {
            "success": "error" not in result,
            **result
        }
    
    def _get_collections(self) -> Dict[str, Any]:
        """Get all available collections."""
        collections = list(self.memory_agent.collections.keys())
        
        return {
            "success": True,
            "collections": collections
        }
    
    def _get_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections."""
        stats = {}
        total = 0
        for collection_name, collection in self.memory_agent.collections.items():
            count = collection.count()
            stats[collection_name] = count
            total += count
        stats["total"] = total
        
        return {
            "success": True,
            "stats": stats
        }
