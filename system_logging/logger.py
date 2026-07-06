from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import json
import threading


class LogLevel(Enum):
    """Log levels for the system."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """A single log entry."""
    timestamp: str
    level: LogLevel
    component: str
    message: str
    context: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert log entry to dictionary."""
        return {
            "timestamp": self.timestamp,
            "level": self.level.value,
            "component": self.component,
            "message": self.message,
            "context": self.context
        }


class SystemLogger:
    """Comprehensive logging system for the AI agent system."""
    
    def __init__(self, log_dir: str = "logs", max_file_size: int = 10 * 1024 * 1024):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.max_file_size = max_file_size
        self._logs: List[LogEntry] = []
        self._max_memory_logs = 10000
        self._lock = threading.Lock()
        self._current_log_file = self._get_current_log_file()
    
    def _get_current_log_file(self) -> Path:
        """Get the current log file path."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"system_{date_str}.log"
    
    def log(self, level: LogLevel, component: str, message: str, 
            context: Optional[Dict[str, Any]] = None) -> None:
        """Log a message with the specified level."""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            component=component,
            message=message,
            context=context or {}
        )
        
        with self._lock:
            self._logs.append(entry)
            
            # Trim memory logs if too large
            if len(self._logs) > self._max_memory_logs:
                self._logs = self._logs[-self._max_memory_logs:]
            
            # Write to file
            self._write_to_file(entry)
    
    def debug(self, component: str, message: str, context: Optional[Dict] = None) -> None:
        """Log a debug message."""
        self.log(LogLevel.DEBUG, component, message, context)
    
    def info(self, component: str, message: str, context: Optional[Dict] = None) -> None:
        """Log an info message."""
        self.log(LogLevel.INFO, component, message, context)
    
    def warning(self, component: str, message: str, context: Optional[Dict] = None) -> None:
        """Log a warning message."""
        self.log(LogLevel.WARNING, component, message, context)
    
    def error(self, component: str, message: str, context: Optional[Dict] = None) -> None:
        """Log an error message."""
        self.log(LogLevel.ERROR, component, message, context)
    
    def critical(self, component: str, message: str, context: Optional[Dict] = None) -> None:
        """Log a critical message."""
        self.log(LogLevel.CRITICAL, component, message, context)
    
    def _write_to_file(self, entry: LogEntry) -> None:
        """Write log entry to file."""
        try:
            # Check if we need to rotate log file
            if self._current_log_file.exists() and self._current_log_file.stat().st_size > self.max_file_size:
                self._current_log_file = self._get_current_log_file()
            
            with open(self._current_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry.to_dict()) + '\n')
        except Exception as e:
            print(f"Error writing to log file: {e}")
    
    def get_logs(self, level: Optional[LogLevel] = None, component: Optional[str] = None,
                 limit: int = 100) -> List[Dict]:
        """Get logs from memory, optionally filtered."""
        with self._lock:
            filtered = self._logs
            
            if level:
                filtered = [log for log in filtered if log.level == level]
            
            if component:
                filtered = [log for log in filtered if log.component == component]
            
            # Get most recent logs
            filtered = filtered[-limit:]
            
            return [log.to_dict() for log in filtered]
    
    def get_logs_from_file(self, date: Optional[str] = None) -> List[Dict]:
        """Get logs from file for a specific date."""
        if date:
            log_file = self.log_dir / f"system_{date}.log"
        else:
            log_file = self._current_log_file
        
        if not log_file.exists():
            return []
        
        try:
            logs = []
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        logs.append(json.loads(line))
            return logs
        except Exception as e:
            print(f"Error reading log file: {e}")
            return []
    
    def clear_memory_logs(self) -> None:
        """Clear logs from memory."""
        with self._lock:
            self._logs.clear()
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get statistics about logs."""
        with self._lock:
            level_counts = {}
            component_counts = {}
            
            for log in self._logs:
                level_counts[log.level.value] = level_counts.get(log.level.value, 0) + 1
                component_counts[log.component] = component_counts.get(log.component, 0) + 1
            
            return {
                "total_logs": len(self._logs),
                "level_counts": level_counts,
                "component_counts": component_counts,
                "current_log_file": str(self._current_log_file),
                "log_file_exists": self._current_log_file.exists(),
                "log_file_size": self._current_log_file.stat().st_size if self._current_log_file.exists() else 0
            }
    
    def search_logs(self, query: str, limit: int = 100) -> List[Dict]:
        """Search logs for a query string."""
        with self._lock:
            query_lower = query.lower()
            filtered = [
                log for log in self._logs
                if query_lower in log.message.lower() or
                query_lower in log.component.lower() or
                any(query_lower in str(v).lower() for v in log.context.values())
            ]
            
            filtered = filtered[-limit:]
            return [log.to_dict() for log in filtered]


# Global logger instance
_global_logger: Optional[SystemLogger] = None


def get_global_logger() -> SystemLogger:
    """Get or create the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = SystemLogger()
    return _global_logger


def reset_global_logger() -> None:
    """Reset the global logger instance."""
    global _global_logger
    _global_logger = None
