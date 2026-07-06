from typing import Dict, List, Optional, Any
from datetime import datetime
from threading import Lock
import json


class SharedState:
    """Shared state system for agents to communicate and coordinate."""
    
    def __init__(self):
        self._state = {
            "goal": None,
            "current_step": 0,
            "total_steps": 0,
            "project": {},
            "memory": [],
            "files": {},
            "results": {},
            "errors": [],
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
        }
        self._lock = Lock()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from shared state."""
        with self._lock:
            keys = key.split(".")
            value = self._state
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in shared state."""
        with self._lock:
            keys = key.split(".")
            current = self._state
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
            self._update_timestamp()
    
    def update(self, key: str, updates: Dict[str, Any]) -> None:
        """Update a nested dictionary in shared state."""
        with self._lock:
            current = self.get(key, {})
            if isinstance(current, dict):
                current.update(updates)
                self.set(key, current)
            self._update_timestamp()
    
    def append(self, key: str, value: Any) -> None:
        """Append a value to a list in shared state."""
        with self._lock:
            current = self.get(key, [])
            if not isinstance(current, list):
                current = []
            current.append(value)
            self.set(key, current)
            self._update_timestamp()
    
    def remove(self, key: str, value: Any) -> None:
        """Remove a value from a list in shared state."""
        with self._lock:
            current = self.get(key, [])
            if isinstance(current, list) and value in current:
                current.remove(value)
                self.set(key, current)
            self._update_timestamp()
    
    def increment(self, key: str, amount: int = 1) -> None:
        """Increment a numeric value in shared state."""
        with self._lock:
            current = self.get(key, 0)
            if isinstance(current, (int, float)):
                self.set(key, current + amount)
            self._update_timestamp()
    
    def get_all(self) -> Dict[str, Any]:
        """Get entire shared state."""
        with self._lock:
            return self._state.copy()
    
    def reset(self) -> None:
        """Reset shared state to initial state."""
        with self._lock:
            self._state = {
                "goal": None,
                "current_step": 0,
                "total_steps": 0,
                "project": {},
                "memory": [],
                "files": {},
                "results": {},
                "errors": [],
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
            }
    
    def initialize_task(self, goal: str, total_steps: int = 0) -> None:
        """Initialize shared state for a new task."""
        with self._lock:
            self._state["goal"] = goal
            self._state["current_step"] = 0
            self._state["total_steps"] = total_steps
            self._state["results"] = {}
            self._state["errors"] = []
            self._update_timestamp()
    
    def add_result(self, step: int, agent: str, result: Any) -> None:
        """Add a result from an agent execution."""
        with self._lock:
            if "results" not in self._state:
                self._state["results"] = {}
            self._state["results"][f"step_{step}"] = {
                "agent": agent,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            self._update_timestamp()
    
    def add_error(self, step: int, agent: str, error: str) -> None:
        """Add an error from an agent execution."""
        with self._lock:
            if "errors" not in self._state:
                self._state["errors"] = []
            self._state["errors"].append({
                "step": step,
                "agent": agent,
                "error": error,
                "timestamp": datetime.now().isoformat()
            })
            self._update_timestamp()
    
    def add_file(self, file_path: str, content: str) -> None:
        """Add a file to shared state."""
        with self._lock:
            if "files" not in self._state:
                self._state["files"] = {}
            self._state["files"][file_path] = {
                "content": content,
                "added_at": datetime.now().isoformat()
            }
            self._update_timestamp()
    
    def add_memory(self, memory: Dict[str, Any]) -> None:
        """Add memory to shared state."""
        with self._lock:
            if "memory" not in self._state:
                self._state["memory"] = []
            self._state["memory"].append({
                **memory,
                "added_at": datetime.now().isoformat()
            })
            self._update_timestamp()
    
    def set_project_info(self, project_info: Dict[str, Any]) -> None:
        """Set project information in shared state."""
        with self._lock:
            self._state["project"] = project_info
            self._update_timestamp()
    
    def advance_step(self) -> int:
        """Advance to the next step and return the new step number."""
        with self._lock:
            self._state["current_step"] += 1
            self._update_timestamp()
            return self._state["current_step"]
    
    def get_current_step(self) -> int:
        """Get current step number."""
        with self._lock:
            return self._state.get("current_step", 0)
    
    def is_complete(self) -> bool:
        """Check if the task is complete."""
        with self._lock:
            return self._state.get("current_step", 0) >= self._state.get("total_steps", 0)
    
    def to_json(self) -> str:
        """Convert shared state to JSON string."""
        with self._lock:
            return json.dumps(self._state, indent=2, default=str)
    
    def from_json(self, json_str: str) -> None:
        """Load shared state from JSON string."""
        with self._lock:
            self._state = json.loads(json_str)
            self._update_timestamp()
    
    def _update_timestamp(self) -> None:
        """Update the timestamp in metadata."""
        self._state["metadata"]["updated_at"] = datetime.now().isoformat()
    
    def get_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the current state."""
        with self._lock:
            return {
                "state": self._state.copy(),
                "timestamp": datetime.now().isoformat()
            }
