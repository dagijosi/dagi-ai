from typing import Dict, List, Callable, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from asyncio import Queue, create_task, gather
import asyncio


@dataclass
class Event:
    """Represents an event in the system."""
    name: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert event to dictionary."""
        return {
            "name": self.name,
            "data": self.data,
            "timestamp": self.timestamp,
            "source": self.source
        }


class EventHandler:
    """Base class for event handlers."""
    
    def __init__(self, name: str):
        self.name = name
    
    async def handle(self, event: Event) -> Optional[Any]:
        """Handle an event. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement handle method")


class EventBus:
    """Event bus for event-driven architecture."""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_queue: Queue = Queue()
        self._event_history: List[Event] = []
        self._max_history_size = 1000
        self._running = False
        self._worker_task = None
    
    def subscribe(self, event_name: str, handler: Callable) -> None:
        """Subscribe to an event."""
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        self._subscribers[event_name].append(handler)
    
    def unsubscribe(self, event_name: str, handler: Callable) -> None:
        """Unsubscribe from an event."""
        if event_name in self._subscribers:
            self._subscribers[event_name] = [h for h in self._subscribers[event_name] if h != handler]
    
    async def emit(self, event_name: str, data: Dict[str, Any], source: Optional[str] = None) -> None:
        """Emit an event."""
        event = Event(name=event_name, data=data, source=source)
        
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history_size:
            self._event_history.pop(0)
        
        # Add to queue for processing
        await self._event_queue.put(event)
    
    async def start(self) -> None:
        """Start the event bus worker."""
        if not self._running:
            self._running = True
            self._worker_task = create_task(self._process_events())
    
    async def stop(self) -> None:
        """Stop the event bus worker."""
        if self._running:
            self._running = False
            if self._worker_task:
                self._worker_task.cancel()
                try:
                    await self._worker_task
                except asyncio.CancelledError:
                    pass
    
    async def _process_events(self) -> None:
        """Process events from the queue."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.1)
                await self._dispatch_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error processing event: {e}")
    
    async def _dispatch_event(self, event: Event) -> None:
        """Dispatch event to all subscribers."""
        handlers = self._subscribers.get(event.name, [])
        
        # Execute all handlers concurrently
        tasks = []
        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                tasks.append(handler(event))
            else:
                # Handle synchronous handlers
                try:
                    result = handler(event)
                    if asyncio.iscoroutine(result):
                        tasks.append(result)
                except Exception as e:
                    print(f"Error in handler: {e}")
        
        if tasks:
            await gather(*tasks, return_exceptions=True)
    
    def get_event_history(self, event_name: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get event history, optionally filtered by event name."""
        history = self._event_history
        
        if event_name:
            history = [e for e in history if e.name == event_name]
        
        # Get most recent events
        history = history[-limit:]
        
        return [e.to_dict() for e in history]
    
    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()
    
    def get_subscribers(self) -> Dict[str, int]:
        """Get count of subscribers for each event."""
        return {
            event_name: len(handlers)
            for event_name, handlers in self._subscribers.items()
        }
    
    def is_running(self) -> bool:
        """Check if event bus is running."""
        return self._running


# Global event bus instance
_global_event_bus: Optional[EventBus] = None


def get_global_event_bus() -> EventBus:
    """Get or create the global event bus instance."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


def reset_global_event_bus() -> None:
    """Reset the global event bus instance."""
    global _global_event_bus
    _global_event_bus = None
