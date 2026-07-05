from typing import Dict, List, Optional, Any
from chromadb import Client, PersistentClient
from chromadb.config import Settings
from datetime import datetime
import uuid


class MemoryAgent:
    """Manages long-term memory with multiple collections and rich metadata for RAG."""
    
    def __init__(self, persist_directory: str = "memory/chroma"):
        self.status = "idle"
        self.client = PersistentClient(path=persist_directory)
        
        # Core collections for different memory types
        self.collections = {
            'documents': self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            ),
            'code': self.client.get_or_create_collection(
                name="code",
                metadata={"hnsw:space": "cosine"}
            ),
            'conversation': self.client.get_or_create_collection(
                name="conversation",
                metadata={"hnsw:space": "cosine"}
            ),
            'notes': self.client.get_or_create_collection(
                name="notes",
                metadata={"hnsw:space": "cosine"}
            ),
            'errors': self.client.get_or_create_collection(
                name="errors",
                metadata={"hnsw:space": "cosine"}
            ),
            'projects': self.client.get_or_create_collection(
                name="projects",
                metadata={"hnsw:space": "cosine"}
            ),
            'corrections': self.client.get_or_create_collection(
                name="corrections",
                metadata={"hnsw:space": "cosine"}
            )
        }
        
        # Backward compatibility
        self.conversation_collection = self.collections['conversation']
        self.document_collection = self.collections['documents']
        self.corrections_collection = self.collections['corrections']
    
    def execute(self, task_data: Dict) -> Dict:
        """Execute a memory-related task."""
        self.status = "working"
        task_type = task_data.get('type')
        
        try:
            if task_type == 'save_memory':
                result = self.save_memory(
                    task_data.get('content'),
                    task_data.get('collection', 'documents'),
                    task_data.get('metadata', {})
                )
            elif task_type == 'search_memory':
                result = self.search_memory(
                    task_data.get('query'),
                    task_data.get('collection', 'documents'),
                    task_data.get('n_results', 5),
                    task_data.get('where', None)
                )
            elif task_type == 'save_conversation':
                result = self.save_conversation(
                    task_data.get('user_message'),
                    task_data.get('assistant_message'),
                    task_data.get('project'),
                    task_data.get('agent'),
                    task_data.get('conversation_id')
                )
            elif task_type == 'search_documents':
                result = self.search_documents(
                    task_data.get('query'),
                    task_data.get('n_results', 5),
                    task_data.get('where', None)
                )
            elif task_type == 'search_code':
                result = self.search_code(
                    task_data.get('query'),
                    task_data.get('n_results', 5),
                    task_data.get('where', None)
                )
            elif task_type == 'delete_memory':
                result = self.delete_memory(
                    task_data.get('memory_id'),
                    task_data.get('collection', 'documents')
                )
            elif task_type == 'update_memory':
                result = self.update_memory(
                    task_data.get('memory_id'),
                    task_data.get('collection', 'documents'),
                    task_data.get('content'),
                    task_data.get('metadata', None)
                )
            # Backward compatibility
            elif task_type == 'store_conversation':
                result = self._store_conversation(task_data)
            elif task_type == 'retrieve_conversation':
                result = self._retrieve_conversation(task_data)
            elif task_type == 'store_document':
                result = self._store_document(task_data)
            elif task_type == 'store_correction':
                result = self._store_correction(task_data)
            elif task_type == 'retrieve_corrections':
                result = self._retrieve_corrections(task_data)
            elif task_type == 'search_corrections':
                result = self._search_corrections(task_data)
            else:
                result = {'error': f'Unknown task type: {task_type}'}
            
            self.status = "idle"
            return result
        except Exception as e:
            self.status = "error"
            return {'error': str(e)}
    
    def save_memory(self, content: str, collection: str = 'documents', metadata: Optional[Dict] = None) -> Dict:
        """Save a memory with rich metadata to a specific collection.
        
        Args:
            content: The text content to store
            collection: Collection name (documents, code, conversation, notes, errors, projects)
            metadata: Dictionary with keys like source, type, project, language, tags, etc.
        
        Returns:
            Dict with status and memory_id
        """
        if collection not in self.collections:
            return {'error': f'Invalid collection: {collection}. Available: {list(self.collections.keys())}'}
        
        if metadata is None:
            metadata = {}
        
        # Add timestamp if not provided
        if 'created_at' not in metadata:
            metadata['created_at'] = datetime.now().isoformat()
        
        # Generate unique ID
        memory_id = str(uuid.uuid4())
        
        self.collections[collection].add(
            documents=[content],
            metadatas=[metadata],
            ids=[memory_id]
        )
        
        return {'status': 'stored', 'memory_id': memory_id, 'collection': collection}
    
    def search_memory(self, query: str, collection: str = 'documents', n_results: int = 5, where: Optional[Dict] = None) -> Dict:
        """Search for relevant memories in a collection with optional metadata filtering.
        
        Args:
            query: Search query text
            collection: Collection name to search
            n_results: Number of results to return
            where: Optional metadata filter (e.g., {"project": "Insurance Portal", "language": "python"})
        
        Returns:
            Dict with results, metadatas, and distances
        """
        if collection not in self.collections:
            return {'error': f'Invalid collection: {collection}. Available: {list(self.collections.keys())}'}
        
        results = self.collections[collection].query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        
        return {
            'status': 'searched',
            'results': results['documents'][0] if results['documents'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'distances': results['distances'][0] if results['distances'] else [],
            'ids': results['ids'][0] if results['ids'] else []
        }
    
    def save_conversation(self, user_message: str, assistant_message: str, 
                          project: Optional[str] = None, agent: Optional[str] = None,
                          conversation_id: Optional[str] = None) -> Dict:
        """Save a complete conversation turn with rich metadata.
        
        Args:
            user_message: The user's message
            assistant_message: The assistant's response
            project: Optional project name
            agent: Optional agent name that handled the request
            conversation_id: Optional conversation ID for grouping
        
        Returns:
            Dict with status and conversation details
        """
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
        
        timestamp = datetime.now().isoformat()
        
        # Store user message
        user_metadata = {
            'role': 'user',
            'conversation_id': conversation_id,
            'project': project,
            'agent': agent,
            'timestamp': timestamp
        }
        
        self.collections['conversation'].add(
            documents=[user_message],
            metadatas=[user_metadata],
            ids=[f"{conversation_id}_user_{timestamp}"]
        )
        
        # Store assistant message
        assistant_metadata = {
            'role': 'assistant',
            'conversation_id': conversation_id,
            'project': project,
            'agent': agent,
            'timestamp': timestamp
        }
        
        self.collections['conversation'].add(
            documents=[assistant_message],
            metadatas=[assistant_metadata],
            ids=[f"{conversation_id}_assistant_{timestamp}"]
        )
        
        return {
            'status': 'stored',
            'conversation_id': conversation_id,
            'project': project,
            'agent': agent,
            'timestamp': timestamp
        }
    
    def search_documents(self, query: str, n_results: int = 5, where: Optional[Dict] = None) -> Dict:
        """Search for relevant documents with optional metadata filtering.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            where: Optional metadata filter (e.g., {"project": "Insurance Portal", "type": "pdf"})
        
        Returns:
            Dict with results, metadatas, and distances
        """
        return self.search_memory(query, 'documents', n_results, where)
    
    def search_code(self, query: str, n_results: int = 5, where: Optional[Dict] = None) -> Dict:
        """Search for relevant code snippets with optional metadata filtering.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            where: Optional metadata filter (e.g., {"language": "python", "project": "ERP"})
        
        Returns:
            Dict with results, metadatas, and distances
        """
        return self.search_memory(query, 'code', n_results, where)
    
    def delete_memory(self, memory_id: str, collection: str = 'documents') -> Dict:
        """Delete a memory from a collection by ID.
        
        Args:
            memory_id: The ID of the memory to delete
            collection: Collection name
        
        Returns:
            Dict with status
        """
        if collection not in self.collections:
            return {'error': f'Invalid collection: {collection}. Available: {list(self.collections.keys())}'}
        
        self.collections[collection].delete(ids=[memory_id])
        
        return {'status': 'deleted', 'memory_id': memory_id, 'collection': collection}
    
    def update_memory(self, memory_id: str, collection: str = 'documents', 
                      content: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict:
        """Update a memory's content and/or metadata.
        
        Args:
            memory_id: The ID of the memory to update
            collection: Collection name
            content: New content (optional)
            metadata: New metadata (optional, will be merged with existing)
        
        Returns:
            Dict with status
        """
        if collection not in self.collections:
            return {'error': f'Invalid collection: {collection}. Available: {list(self.collections.keys())}'}
        
        # Get existing memory
        existing = self.collections[collection].get(ids=[memory_id])
        
        if not existing['ids']:
            return {'error': f'Memory ID {memory_id} not found in collection {collection}'}
        
        # Delete old memory
        self.collections[collection].delete(ids=[memory_id])
        
        # Merge metadata
        existing_metadata = existing['metadatas'][0] if existing['metadatas'] else {}
        if metadata:
            existing_metadata.update(metadata)
        
        # Add updated timestamp
        existing_metadata['updated_at'] = datetime.now().isoformat()
        
        # Use new content or existing
        new_content = content if content is not None else existing['documents'][0]
        
        # Re-add with updated data
        self.collections[collection].add(
            documents=[new_content],
            metadatas=[existing_metadata],
            ids=[memory_id]
        )
        
        return {'status': 'updated', 'memory_id': memory_id, 'collection': collection}
    
    # Backward compatibility methods
    def _store_conversation(self, task_data: Dict) -> Dict:
        """Store conversation turn in memory (legacy method)."""
        conversation_id = task_data.get('conversation_id')
        message = task_data.get('message')
        role = task_data.get('role', 'user')
        
        self.conversation_collection.add(
            documents=[message],
            metadatas=[{"role": role, "conversation_id": conversation_id}],
            ids=[f"{conversation_id}_{len(self.conversation_collection.get()['ids'])}"]
        )
        
        return {'status': 'stored', 'conversation_id': conversation_id}
    
    def _retrieve_conversation(self, task_data: Dict) -> Dict:
        """Retrieve conversation history (legacy method)."""
        conversation_id = task_data.get('conversation_id')
        limit = task_data.get('limit', 10)
        
        results = self.conversation_collection.get(
            where={"conversation_id": conversation_id},
            limit=limit
        )
        
        return {
            'status': 'retrieved',
            'messages': results['documents'],
            'metadatas': results['metadatas']
        }
    
    def _store_document(self, task_data: Dict) -> Dict:
        """Store document embedding (legacy method)."""
        document_id = task_data.get('document_id')
        content = task_data.get('content')
        metadata = task_data.get('metadata', {})
        
        self.document_collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[document_id]
        )
        
        return {'status': 'stored', 'document_id': document_id}
    
    
    def _store_correction(self, task_data: Dict) -> Dict:
        """Store a correction/feedback for learning (legacy method)."""
        original_response = task_data.get('original_response')
        correction = task_data.get('correction')
        context = task_data.get('context', '')
        agent_type = task_data.get('agent_type', 'unknown')
        
        correction_id = f"correction_{len(self.corrections_collection.get()['ids'])}"
        
        self.corrections_collection.add(
            documents=[f"Original: {original_response}\nCorrection: {correction}\nContext: {context}"],
            metadatas=[{
                "agent_type": agent_type,
                "context": context,
                "original_response": original_response,
                "correction": correction,
                "timestamp": str(self._get_timestamp())
            }],
            ids=[correction_id]
        )
        
        return {'status': 'stored', 'correction_id': correction_id}
    
    def _retrieve_corrections(self, task_data: Dict) -> Dict:
        """Retrieve corrections by agent type or context (legacy method)."""
        agent_type = task_data.get('agent_type')
        limit = task_data.get('limit', 10)
        
        where = {"agent_type": agent_type} if agent_type else None
        
        results = self.corrections_collection.get(
            where=where,
            limit=limit
        )
        
        return {
            'status': 'retrieved',
            'corrections': results['documents'],
            'metadatas': results['metadatas']
        }
    
    def _search_corrections(self, task_data: Dict) -> Dict:
        """Search for relevant corrections based on query (legacy method)."""
        query = task_data.get('query')
        n_results = task_data.get('n_results', 5)
        
        results = self.corrections_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return {
            'status': 'searched',
            'results': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0]
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_status(self) -> str:
        """Get current agent status."""
        return self.status
