from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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
    agent_type: str
    task_type: str
    task_data: Dict


class PlanRequest(BaseModel):
    goal: str
    context: Optional[Dict] = {}


class FileRequest(BaseModel):
    operation: str
    file_path: Optional[str] = None
    content: Optional[str] = None
    dir_path: Optional[str] = None
    pattern: Optional[str] = None
    search_path: Optional[str] = None


class TerminalRequest(BaseModel):
    command: str
    timeout: Optional[int] = 30


class SearchRequest(BaseModel):
    operation: str
    pattern: str
    file_pattern: Optional[str] = "*"
    file_path: Optional[str] = None
    case_sensitive: Optional[bool] = False
    search_path: Optional[str] = "."


class GitRequest(BaseModel):
    operation: str
    files: Optional[str] = "."
    message: Optional[str] = None
    remote: Optional[str] = "origin"
    branch: Optional[str] = "main"
    branch_name: Optional[str] = None
    create_new: Optional[bool] = False
    target: Optional[str] = None
    limit: Optional[int] = 10


class ChatRequest(BaseModel):
    messages: List[Dict] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class ManagerRequest(BaseModel):
    user_input: str


class CorrectionRequest(BaseModel):
    original_response: str
    correction: str
    context: Optional[str] = ""
    agent_type: Optional[str] = "unknown"


class DocQuestionRequest(BaseModel):
    question: str
    doc_path: Optional[str] = None
    doc_content: Optional[str] = None


# Health Check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "dagi-ai"}


# Agent Endpoints
@app.post("/agents/execute")
async def execute_agent_task(request: TaskRequest):
    """Execute a task on a specific agent."""
    try:
        result = agent_manager.route_task(request.agent_type, request.task_data)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/plan")
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


@app.post("/agents/execute-plan")
async def execute_plan(plan: List[Dict]):
    """Execute a multi-step plan."""
    try:
        results = agent_manager.execute_plan(plan)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents/status")
async def get_agent_status():
    """Get status of all agents."""
    return agent_manager.get_agent_status()


@app.post("/agents/chat")
async def chat_with_manager(request: ManagerRequest):
    """Chat with the manager agent using LLM."""
    try:
        response = agent_manager.run(request.user_input)
        return {"status": "success", "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm/chat")
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


@app.get("/llm/models")
async def get_available_models():
    """Get available models from LM Studio."""
    try:
        models = lmstudio_client.get_models()
        return {"status": "success", "models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/llm/health")
async def llm_health_check():
    """Check if LM Studio API is accessible."""
    try:
        healthy = lmstudio_client.health_check()
        return {"status": "healthy" if healthy else "unhealthy"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/correction")
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


@app.get("/memory/corrections")
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


@app.post("/docs/question")
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
@app.post("/tools/file")
async def file_operation(request: FileRequest):
    """Execute file system operations."""
    try:
        task_data = request.dict(exclude_none=True)
        result = file_tool.execute(task_data)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/terminal")
async def terminal_operation(request: TerminalRequest):
    """Execute terminal commands."""
    try:
        task_data = request.dict()
        result = terminal_tool.execute(task_data)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/search")
async def search_operation(request: SearchRequest):
    """Execute search operations."""
    try:
        task_data = request.dict(exclude_none=True)
        result = search_tool.execute(task_data)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/git")
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
