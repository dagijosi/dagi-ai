from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import AgentManager, ManagerAgent
from tools.file_tool import FileTool
from tools.terminal_tool import TerminalTool
from tools.search_tool import SearchTool
from tools.git_tool import GitTool
from api.lmstudio_client import get_lmstudio_client


app = FastAPI(title="Dagi-AI API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents and tools
agent_manager = ManagerAgent()
file_tool = FileTool()
terminal_tool = TerminalTool()
search_tool = SearchTool()
git_tool = GitTool()
lmstudio_client = get_lmstudio_client()


# Request Models
class TaskRequest(BaseModel):
    """Request model for executing agent tasks."""
    agent_type: str = Field(
        ..., 
        description="Type of agent to execute the task",
        example="code"
    )
    task_type: str = Field(
        ..., 
        description="Type of task to perform",
        example="analyze"
    )
    task_data: Dict = Field(
        ..., 
        description="Data required for the task execution",
        example={
            "code": "def hello():\n    print('Hello World')",
            "language": "python",
            "question": "What does this function do?"
        }
    )


class PlanRequest(BaseModel):
    """Request model for creating execution plans."""
    goal: str = Field(
        ..., 
        description="The goal to achieve",
        example="Create a REST API for user management"
    )
    context: Optional[Dict] = Field(
        default={}, 
        description="Additional context for planning",
        example={
            "project": "User Management System",
            "tech_stack": ["FastAPI", "PostgreSQL"]
        }
    )


class FileRequest(BaseModel):
    """Request model for file system operations."""
    operation: str = Field(
        ..., 
        description="Type of file operation",
        example="read"
    )
    file_path: Optional[str] = Field(
        default=None, 
        description="Path to the file",
        example="src/main.py"
    )
    content: Optional[str] = Field(
        default=None, 
        description="Content to write to file",
        example="print('Hello World')"
    )
    dir_path: Optional[str] = Field(
        default=None, 
        description="Directory path for operations",
        example="src/"
    )
    pattern: Optional[str] = Field(
        default=None, 
        description="File pattern for search",
        example="*.py"
    )
    search_path: Optional[str] = Field(
        default=None, 
        description="Path to search in",
        example="./"
    )


class TerminalRequest(BaseModel):
    """Request model for terminal command execution."""
    command: str = Field(
        ..., 
        description="Command to execute",
        example="ls -la"
    )
    timeout: Optional[int] = Field(
        default=30, 
        description="Command timeout in seconds",
        example=30
    )


class SearchRequest(BaseModel):
    """Request model for search operations."""
    operation: str = Field(
        ..., 
        description="Type of search operation",
        example="grep"
    )
    pattern: str = Field(
        ..., 
        description="Search pattern",
        example="def hello"
    )
    file_pattern: Optional[str] = Field(
        default="*", 
        description="File pattern to search in",
        example="*.py"
    )
    file_path: Optional[str] = Field(
        default=None, 
        description="Specific file to search",
        example="src/main.py"
    )
    case_sensitive: Optional[bool] = Field(
        default=False, 
        description="Case sensitive search",
        example=False
    )
    search_path: Optional[str] = Field(
        default=".", 
        description="Directory to search in",
        example="./src"
    )


class GitRequest(BaseModel):
    """Request model for Git operations."""
    operation: str = Field(
        ..., 
        description="Git operation to perform",
        example="commit"
    )
    files: Optional[str] = Field(
        default=".", 
        description="Files to operate on",
        example="."
    )
    message: Optional[str] = Field(
        default=None, 
        description="Commit message",
        example="Add new feature"
    )
    remote: Optional[str] = Field(
        default="origin", 
        description="Git remote",
        example="origin"
    )
    branch: Optional[str] = Field(
        default="main", 
        description="Branch name",
        example="main"
    )
    branch_name: Optional[str] = Field(
        default=None, 
        description="New branch name",
        example="feature/new-feature"
    )
    create_new: Optional[bool] = Field(
        default=False, 
        description="Create new branch",
        example=True
    )
    target: Optional[str] = Field(
        default=None, 
        description="Target for checkout/merge",
        example="main"
    )
    limit: Optional[int] = Field(
        default=10, 
        description="Limit for log operations",
        example=10
    )


class ChatRequest(BaseModel):
    """Request model for LLM chat."""
    messages: List[Dict] = Field(
        default=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ], 
        description="Chat messages",
        example=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"}
        ]
    )
    temperature: Optional[float] = Field(
        default=0.7, 
        description="Sampling temperature",
        example=0.7
    )
    max_tokens: Optional[int] = Field(
        default=None, 
        description="Maximum tokens to generate",
        example=500
    )
    stream: Optional[bool] = Field(
        default=False, 
        description="Enable streaming response",
        example=False
    )


class ManagerRequest(BaseModel):
    """Request model for manager agent chat."""
    user_input: str = Field(
        ..., 
        description="User input for the manager agent",
        example="Write a React login page with form validation"
    )


class CorrectionRequest(BaseModel):
    """Request model for storing corrections."""
    original_response: str = Field(
        ..., 
        description="Original incorrect response",
        example="The function should use print instead of return"
    )
    correction: str = Field(
        ..., 
        description="Corrected response",
        example="The function should use return instead of print"
    )
    context: Optional[str] = Field(
        default="", 
        description="Context of the correction",
        example="Code review for login function"
    )
    agent_type: Optional[str] = Field(
        default="unknown", 
        description="Type of agent that made the error",
        example="code"
    )


class DocQuestionRequest(BaseModel):
    """Request model for documentation questions."""
    question: str = Field(
        ..., 
        description="Question about documentation",
        example="How do I authenticate users in this API?"
    )
    doc_path: Optional[str] = Field(
        default=None, 
        description="Path to documentation file",
        example="docs/api.md"
    )
    doc_content: Optional[str] = Field(
        default=None, 
        description="Documentation content",
        example="# API Documentation\n## Authentication\nUse JWT tokens..."
    )


class MemorySaveRequest(BaseModel):
    """Request model for saving memories."""
    content: str = Field(
        ..., 
        description="Content to store in memory",
        example="React hooks allow you to use state and other React features without writing a class."
    )
    collection: str = Field(
        default="documents", 
        description="Collection name (documents, code, conversation, notes, errors, projects)",
        example="documents"
    )
    metadata: Optional[Dict] = Field(
        default={}, 
        description="Metadata for the memory (source, type, project, language, tags, etc.)",
        example={
            "source": "ReactGuide.pdf",
            "type": "document",
            "project": "Insurance Portal",
            "language": "javascript",
            "tags": ["react", "hooks"]
        }
    )


class MemorySearchRequest(BaseModel):
    """Request model for searching memories."""
    query: str = Field(
        ..., 
        description="Search query text",
        example="How to use React hooks"
    )
    collection: str = Field(
        default="documents", 
        description="Collection name to search",
        example="documents"
    )
    n_results: int = Field(
        default=5, 
        description="Number of results to return",
        example=5
    )
    where: Optional[Dict] = Field(
        default=None, 
        description="Metadata filter for search",
        example={"project": "Insurance Portal", "language": "javascript"}
    )


class MemoryDeleteRequest(BaseModel):
    """Request model for deleting memories."""
    memory_id: str = Field(
        ..., 
        description="ID of the memory to delete",
        example="abc123-def456-ghi789"
    )
    collection: str = Field(
        default="documents", 
        description="Collection name",
        example="documents"
    )


class MemoryUpdateRequest(BaseModel):
    """Request model for updating memories."""
    memory_id: str = Field(
        ..., 
        description="ID of the memory to update",
        example="abc123-def456-ghi789"
    )
    collection: str = Field(
        default="documents", 
        description="Collection name",
        example="documents"
    )
    content: Optional[str] = Field(
        default=None, 
        description="New content (optional)",
        example="Updated content about React hooks"
    )
    metadata: Optional[Dict] = Field(
        default=None, 
        description="New metadata to merge (optional)",
        example={"tags": ["react", "hooks", "updated"]}
    )


class ConversationSaveRequest(BaseModel):
    """Request model for saving conversations."""
    user_message: str = Field(
        ..., 
        description="User's message",
        example="How do I create a login form?"
    )
    assistant_message: str = Field(
        ..., 
        description="Assistant's response",
        example="To create a login form in React, you'll need..."
    )
    project: Optional[str] = Field(
        default=None, 
        description="Project name",
        example="Insurance Portal"
    )
    agent: Optional[str] = Field(
        default=None, 
        description="Agent that handled the request",
        example="react"
    )
    conversation_id: Optional[str] = Field(
        default=None, 
        description="Conversation ID for grouping",
        example="conv-123"
    )


# Response Models
class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(
        ..., 
        description="Service health status",
        example="healthy"
    )
    service: str = Field(
        ..., 
        description="Service name",
        example="dagi-ai"
    )


class AgentStatusResponse(BaseModel):
    """Response model for agent status."""
    code: str = Field(
        ..., 
        description="Code agent status",
        example="idle"
    )
    memory: str = Field(
        ..., 
        description="Memory agent status",
        example="idle"
    )
    docs: str = Field(
        ..., 
        description="Docs agent status",
        example="idle"
    )
    planner: str = Field(
        ..., 
        description="Planner agent status",
        example="idle"
    )
    general: str = Field(
        ..., 
        description="General agent status",
        example="idle"
    )


class SuccessResponse(BaseModel):
    """Generic success response model."""
    status: str = Field(
        ..., 
        description="Operation status",
        example="success"
    )
    result: Dict = Field(
        ..., 
        description="Operation result",
        example={"message": "Operation completed successfully"}
    )


class MemorySaveResponse(BaseModel):
    """Response model for memory save operations."""
    status: str = Field(
        ..., 
        description="Operation status",
        example="stored"
    )
    memory_id: str = Field(
        ..., 
        description="ID of the saved memory",
        example="abc123-def456-ghi789"
    )
    collection: str = Field(
        ..., 
        description="Collection name",
        example="documents"
    )


class MemorySearchResponse(BaseModel):
    """Response model for memory search operations."""
    status: str = Field(
        ..., 
        description="Operation status",
        example="searched"
    )
    results: List[str] = Field(
        ..., 
        description="Search results (content)",
        example=["React hooks allow you to use state...", "useState is a hook that lets you add state..."]
    )
    metadatas: List[Dict] = Field(
        ..., 
        description="Metadata for each result",
        example=[{"source": "ReactGuide.pdf", "project": "Insurance Portal"}]
    )
    distances: List[float] = Field(
        ..., 
        description="Distance scores for each result",
        example=[0.123, 0.456]
    )
    ids: List[str] = Field(
        ..., 
        description="IDs of the results",
        example=["abc123", "def456"]
    )


class ConversationSaveResponse(BaseModel):
    """Response model for conversation save operations."""
    status: str = Field(
        ..., 
        description="Operation status",
        example="stored"
    )
    conversation_id: str = Field(
        ..., 
        description="Conversation ID",
        example="conv-123"
    )
    project: Optional[str] = Field(
        default=None, 
        description="Project name",
        example="Insurance Portal"
    )
    agent: Optional[str] = Field(
        default=None, 
        description="Agent name",
        example="react"
    )
    timestamp: str = Field(
        ..., 
        description="Timestamp of the conversation",
        example="2026-07-05T12:00:00"
    )


# Health Check
@app.get("/health", response_model=HealthResponse, responses={
    200: {
        "description": "Service is healthy",
        "content": {
            "application/json": {
                "example": {
                    "status": "healthy",
                    "service": "dagi-ai"
                }
            }
        }
    }
})
async def health_check():
    """Check if the API service is running."""
    return {"status": "healthy", "service": "dagi-ai"}


# Agent Endpoints
@app.post("/agents/execute", response_model=SuccessResponse, responses={
    200: {
        "description": "Task executed successfully",
        "content": {
            "application/json": {
                "example": {
                    "status": "success",
                    "result": {
                        "analysis": "Code analyzed successfully",
                        "suggestions": ["Add error handling", "Improve variable names"]
                    }
                }
            }
        }
    },
    500: {
        "description": "Error executing task",
        "content": {
            "application/json": {
                "example": {
                    "detail": "Agent not found: invalid_agent"
                }
            }
        }
    }
})
async def execute_agent_task(request: TaskRequest):
    """Execute a task on a specific agent."""
    try:
        # Add task_type to task_data for agent execution
        task_data_with_type = request.task_data.copy()
        task_data_with_type['type'] = request.task_type
        result = agent_manager.route_task(request.agent_type, task_data_with_type)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/plan", response_model=SuccessResponse, responses={
    200: {
        "description": "Plan created successfully",
        "content": {
            "application/json": {
                "example": {
                    "status": "success",
                    "result": {
                        "plan": [
                            {"step": 1, "task": "Design database schema", "agent": "database_expert"},
                            {"step": 2, "task": "Implement API endpoints", "agent": "api"},
                            {"step": 3, "task": "Write tests", "agent": "qa_tester"}
                        ]
                    }
                }
            }
        }
    }
})
async def create_plan(request: PlanRequest):
    """Create a plan for a complex task."""
    try:
        task_data = {
            "type": "create_plan",
            "goal": request.goal,
            "context": request.context
        }
        result = agent_manager.planner_agent.execute(task_data)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/execute-plan", response_model=SuccessResponse, responses={
    200: {
        "description": "Plan executed successfully",
        "content": {
            "application/json": {
                "example": {
                    "status": "success",
                    "results": [
                        {"step": 1, "agent": "database_expert", "result": "Schema created"},
                        {"step": 2, "agent": "api", "result": "Endpoints implemented"},
                        {"step": 3, "agent": "qa_tester", "result": "Tests passed"}
                    ]
                }
            }
        }
    }
})
async def execute_plan(plan: List[Dict]):
    """Execute a multi-step plan."""
    try:
        results = agent_manager.execute_plan(plan)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents/status", response_model=AgentStatusResponse, responses={
    200: {
        "description": "Agent statuses retrieved",
        "content": {
            "application/json": {
                "example": {
                    "code": "idle",
                    "memory": "idle",
                    "docs": "idle",
                    "planner": "idle",
                    "general": "idle",
                    "ui_designer": "idle",
                    "qa_tester": "idle",
                    "security_auditor": "idle",
                    "database_expert": "idle",
                    "devops": "idle",
                    "api": "idle",
                    "flutter": "idle",
                    "react": "idle"
                }
            }
        }
    }
})
async def get_agent_status():
    """Get status of all agents."""
    return agent_manager.get_agent_status()


@app.post("/agents/chat", response_model=SuccessResponse, responses={
    200: {
        "description": "Chat response generated",
        "content": {
            "application/json": {
                "example": {
                    "status": "success",
                    "response": {
                        "manager_analysis": {
                            "analysis": "User wants to create a React login page",
                            "complexity": "medium",
                            "total_steps": 3
                        },
                        "execution_plan": [
                            {"step": 1, "agent": "ui_designer", "task": "Design login page layout"},
                            {"step": 2, "agent": "react", "task": "Implement React components"},
                            {"step": 3, "agent": "code", "task": "Review and optimize code"}
                        ],
                        "final_product": {
                            "summary": "React login page created with form validation",
                            "code": "// React component code here"
                        }
                    }
                }
            }
        }
    }
})
async def chat_with_manager(request: ManagerRequest):
    """Chat with the manager agent using LLM."""
    try:
        response = agent_manager.run(request.user_input)
        return {"status": "success", "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm/chat", response_model=SuccessResponse, responses={
    200: {
        "description": "LLM response generated",
        "content": {
            "application/json": {
                "example": {
                    "status": "success",
                    "result": {
                        "message": "Python is a high-level programming language known for its simplicity and readability.",
                        "model": "llama-3-8b",
                        "tokens_used": 45
                    }
                }
            }
        }
    }
})
async def chat_with_llm(request: ChatRequest):
    """Direct chat with LLM."""
    try:
        if request.stream:
            # Streaming response
            from fastapi.responses import StreamingResponse
            
            def generate():
                for chunk in lmstudio_client.chat_stream(
                    messages=request.messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                ):
                    yield f"data: {chunk}\n\n"
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            result = lmstudio_client.chat(
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False
            )
            return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/llm/models", responses={
    200: {
        "description": "Available models retrieved",
        "content": {
            "application/json": {
                "example": {
                    "status": "success",
                    "models": [
                        {"id": "llama-3-8b", "name": "Llama 3 8B", "size": "4.7GB"},
                        {"id": "llama-3-70b", "name": "Llama 3 70B", "size": "40GB"},
                        {"id": "mistral-7b", "name": "Mistral 7B", "size": "4.1GB"}
                    ]
                }
            }
        }
    }
})
async def get_available_models():
    """Get available models from LM Studio."""
    try:
        models = lmstudio_client.get_models()
        return {"status": "success", "models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/llm/health", responses={
    200: {
        "description": "LM Studio health status",
        "content": {
            "application/json": {
                "example": {"status": "healthy"}
            }
        }
    }
})
async def llm_health_check():
    """Check if LM Studio API is accessible."""
    try:
        healthy = lmstudio_client.health_check()
        return {"status": "healthy" if healthy else "unhealthy"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/correction", response_model=SuccessResponse, responses={
    200: {
        "description": "Correction stored successfully",
        "content": {
            "application/json": {
                "example": {
                    "status": "success",
                    "result": {
                        "status": "stored",
                        "correction_id": "correction_123",
                        "timestamp": "2026-07-05T12:00:00"
                    }
                }
            }
        }
    }
})
async def store_correction(request: CorrectionRequest):
    """Store a correction in memory."""
    try:
        task_data = {
            "type": "store_correction",
            "original_response": request.original_response,
            "correction": request.correction,
            "context": request.context,
            "agent_type": request.agent_type
        }
        result = agent_manager.memory_agent.execute(task_data)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/corrections", response_model=SuccessResponse, responses={
    200: {
        "description": "Corrections retrieved",
        "content": {
            "application/json": {
                "example": {
                    "status": "success",
                    "result": {
                        "status": "retrieved",
                        "corrections": [
                            "Original: Use print\nCorrection: Use return\nContext: Function design"
                        ],
                        "metadatas": [
                            {"agent_type": "code", "timestamp": "2026-07-05T12:00:00"}
                        ]
                    }
                }
            }
        }
    }
})
async def get_corrections(agent_type: Optional[str] = None, limit: int = 10):
    """Retrieve corrections from memory."""
    try:
        task_data = {
            "type": "retrieve_corrections",
            "agent_type": agent_type,
            "limit": limit
        }
        result = agent_manager.memory_agent.execute(task_data)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Enhanced Memory Endpoints
@app.post("/memory/save", response_model=MemorySaveResponse, responses={
    200: {
        "description": "Memory saved successfully",
        "content": {
            "application/json": {
                "example": {
                    "status": "stored",
                    "memory_id": "abc123-def456-ghi789",
                    "collection": "documents"
                }
            }
        }
    }
})
async def save_memory(request: MemorySaveRequest):
    """Save a memory with rich metadata to a specific collection."""
    try:
        result = agent_manager.memory_agent.save_memory(
            content=request.content,
            collection=request.collection,
            metadata=request.metadata
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/search", response_model=MemorySearchResponse, responses={
    200: {
        "description": "Memory search completed",
        "content": {
            "application/json": {
                "example": {
                    "status": "searched",
                    "results": [
                        "React hooks allow you to use state and other React features without writing a class.",
                        "useState is a hook that lets you add React state to function components."
                    ],
                    "metadatas": [
                        {"source": "ReactGuide.pdf", "project": "Insurance Portal"},
                        {"source": "ReactDocs.md", "project": "Insurance Portal"}
                    ],
                    "distances": [0.123, 0.456],
                    "ids": ["abc123", "def456"]
                }
            }
        }
    }
})
async def search_memory(request: MemorySearchRequest):
    """Search for relevant memories in a collection with optional metadata filtering."""
    try:
        result = agent_manager.memory_agent.search_memory(
            query=request.query,
            collection=request.collection,
            n_results=request.n_results,
            where=request.where
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/delete", response_model=SuccessResponse, responses={
    200: {
        "description": "Memory deleted successfully",
        "content": {
            "application/json": {
                "example": {
                    "status": "success",
                    "result": {
                        "status": "deleted",
                        "memory_id": "abc123-def456-ghi789",
                        "collection": "documents"
                    }
                }
            }
        }
    }
})
async def delete_memory(request: MemoryDeleteRequest):
    """Delete a memory from a collection by ID."""
    try:
        result = agent_manager.memory_agent.delete_memory(
            memory_id=request.memory_id,
            collection=request.collection
        )
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/memory/update", response_model=SuccessResponse, responses={
    200: {
        "description": "Memory updated successfully",
        "content": {
            "application/json": {
                "example": {
                    "status": "success",
                    "result": {
                        "status": "updated",
                        "memory_id": "abc123-def456-ghi789",
                        "collection": "documents"
                    }
                }
            }
        }
    }
})
async def update_memory(request: MemoryUpdateRequest):
    """Update a memory's content and/or metadata."""
    try:
        result = agent_manager.memory_agent.update_memory(
            memory_id=request.memory_id,
            collection=request.collection,
            content=request.content,
            metadata=request.metadata
        )
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/conversation", response_model=ConversationSaveResponse, responses={
    200: {
        "description": "Conversation saved successfully",
        "content": {
            "application/json": {
                "example": {
                    "status": "stored",
                    "conversation_id": "conv-123",
                    "project": "Insurance Portal",
                    "agent": "react",
                    "timestamp": "2026-07-05T12:00:00"
                }
            }
        }
    }
})
async def save_conversation(request: ConversationSaveRequest):
    """Save a complete conversation turn with rich metadata."""
    try:
        result = agent_manager.memory_agent.save_conversation(
            user_message=request.user_message,
            assistant_message=request.assistant_message,
            project=request.project,
            agent=request.agent,
            conversation_id=request.conversation_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/collections", responses={
    200: {
        "description": "Available collections retrieved",
        "content": {
            "application/json": {
                "example": {
                    "status": "success",
                    "collections": ["documents", "code", "conversation", "notes", "errors", "projects", "corrections"]
                }
            }
        }
    }
})
async def get_collections():
    """Get all available memory collections."""
    try:
        collections = list(agent_manager.memory_agent.collections.keys())
        return {"status": "success", "collections": collections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/stats", responses={
    200: {
        "description": "Memory statistics retrieved",
        "content": {
            "application/json": {
                "example": {
                    "status": "success",
                    "stats": {
                        "documents": 150,
                        "code": 75,
                        "conversation": 200,
                        "notes": 30,
                        "errors": 10,
                        "projects": 5,
                        "corrections": 20,
                        "total": 490
                    }
                }
            }
        }
    }
})
async def get_memory_stats():
    """Get statistics for all memory collections."""
    try:
        stats = {}
        total = 0
        for collection_name, collection in agent_manager.memory_agent.collections.items():
            count = collection.count()
            stats[collection_name] = count
            total += count
        stats["total"] = total
        return {"status": "success", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/docs/question", response_model=SuccessResponse, responses={
    200: {
        "description": "Documentation question answered",
        "content": {
            "application/json": {
                "example": {
                    "status": "success",
                    "result": {
                        "answer": "To authenticate users, use JWT tokens. Include the token in the Authorization header as 'Bearer <token>'.",
                        "relevant_sections": ["Authentication", "JWT Setup"],
                        "confidence": 0.95
                    }
                }
            }
        }
    }
})
async def ask_documentation_question(request: DocQuestionRequest):
    """Ask a question about documentation."""
    try:
        task_data = {
            "type": "answer_question",
            "question": request.question,
            "doc_path": request.doc_path,
            "doc_content": request.doc_content
        }
        result = agent_manager.docs_agent.execute(task_data)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Tool Endpoints
@app.post("/tools/file", response_model=SuccessResponse, responses={
    200: {
        "description": "File operation completed",
        "content": {
            "application/json": {
                "example": {
                    "status": "success",
                    "result": {
                        "operation": "read",
                        "file_path": "src/main.py",
                        "content": "print('Hello World')",
                        "size": 20
                    }
                }
            }
        }
    }
})
async def file_operation(request: FileRequest):
    """Execute file system operations."""
    try:
        task_data = request.dict(exclude_none=True)
        result = file_tool.execute(task_data)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/terminal", response_model=SuccessResponse, responses={
    200: {
        "description": "Terminal command executed",
        "content": {
            "application/json": {
                "example": {
                    "status": "success",
                    "result": {
                        "command": "ls -la",
                        "exit_code": 0,
                        "output": "drwxr-xr-x  2 user user 4096 Jul  5 12:00 src",
                        "stderr": ""
                    }
                }
            }
        }
    }
})
async def terminal_operation(request: TerminalRequest):
    """Execute terminal commands."""
    try:
        task_data = request.dict()
        result = terminal_tool.execute(task_data)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/search", response_model=SuccessResponse, responses={
    200: {
        "description": "Search operation completed",
        "content": {
            "application/json": {
                "example": {
                    "status": "success",
                    "result": {
                        "operation": "grep",
                        "pattern": "def hello",
                        "matches": [
                            {"file": "src/main.py", "line": 10, "content": "def hello():"},
                            {"file": "src/utils.py", "line": 5, "content": "def hello_world():"}
                        ],
                        "total_matches": 2
                    }
                }
            }
        }
    }
})
async def search_operation(request: SearchRequest):
    """Execute search operations."""
    try:
        task_data = request.dict(exclude_none=True)
        result = search_tool.execute(task_data)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/git", response_model=SuccessResponse, responses={
    200: {
        "description": "Git operation completed",
        "content": {
            "application/json": {
                "example": {
                    "status": "success",
                    "result": {
                        "operation": "commit",
                        "commit_hash": "abc123def456",
                        "message": "Add new feature",
                        "files_changed": 3
                    }
                }
            }
        }
    }
})
async def git_operation(request: GitRequest):
    """Execute Git operations."""
    try:
        task_data = request.dict(exclude_none=True)
        result = git_tool.execute(task_data)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
