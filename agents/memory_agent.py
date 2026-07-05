from typing import Dict, List, Optional
from chromadb import Client, PersistentClient
from chromadb.config import Settings


class MemoryAgent:
    """Manages conversation history and document embeddings."""
    
    def __init__(self, persist_directory: str = "memory/chroma"):
        self.status = "idle"
        self.client = PersistentClient(path=persist_directory)
        self.conversation_collection = self.client.get_or_create_collection(
            name="conversations",
            metadata={"hnsw:space": "cosine"}
        )
        self.document_collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        self.corrections_collection = self.client.get_or_create_collection(
            name="corrections",
            metadata={"hnsw:space": "cosine"}
        )
    
    def execute(self, task_data: Dict) -> Dict:
        """Execute a memory-related task."""
        self.status = "working"
        task_type = task_data.get('type')
        
        try:
            if task_type == 'store_conversation':
                result = self._store_conversation(task_data)
            elif task_type == 'retrieve_conversation':
                result = self._retrieve_conversation(task_data)
            elif task_type == 'store_document':
                result = self._store_document(task_data)
            elif task_type == 'search_documents':
                result = self._search_documents(task_data)
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
    
    def _store_conversation(self, task_data: Dict) -> Dict:
        """Store conversation turn in memory."""
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
        """Retrieve conversation history."""
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
        """Store document embedding."""
        document_id = task_data.get('document_id')
        content = task_data.get('content')
        metadata = task_data.get('metadata', {})
        
        self.document_collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[document_id]
        )
        
        return {'status': 'stored', 'document_id': document_id}
    
    def _search_documents(self, task_data: Dict) -> Dict:
        """Search for relevant documents."""
        query = task_data.get('query')
        n_results = task_data.get('n_results', 5)
        
        results = self.document_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return {
            'status': 'searched',
            'results': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0]
        }
    
    def _store_correction(self, task_data: Dict) -> Dict:
        """Store a correction/feedback for learning."""
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
        """Retrieve corrections by agent type or context."""
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
        """Search for relevant corrections based on query."""
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
