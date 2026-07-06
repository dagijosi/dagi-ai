"""
Test script for the new architecture components.

This script tests the planner-executor architecture with:
- Planner: Creates task graphs
- Executor: Executes task graphs
- ContextManager: Builds context for tasks
- SharedState: Manages shared state
- EventBus: Event-driven communication
- AgentRegistry: Dynamic agent registration
- SystemLogger: Comprehensive logging
"""

import asyncio
from planner.planner import Planner, TaskGraph
from executor.executor import Executor
from context.manager import ContextManager, ContextRequest
from state.shared_state import SharedState
from events.event_bus import EventBus, get_global_event_bus
from registry.agent_registry import AgentRegistry
from system_logging.logger import SystemLogger, get_global_logger


def test_planner():
    """Test the Planner component."""
    print("=== Testing Planner ===")
    planner = Planner()
    
    # Create a plan for a simple task
    task_graph = planner.create_plan("Create a login page using React")
    
    print(f"Goal: {task_graph.goal}")
    print(f"Steps: {len(task_graph.steps)}")
    for step in task_graph.steps:
        print(f"  Step {step.step}: {step.agent} - {step.action}")
    
    # Test finding suitable agents
    suitable = planner.get_suitable_agents("generate_code")
    print(f"Suitable agents for 'generate_code': {suitable}")
    
    return task_graph


def test_shared_state():
    """Test the SharedState component."""
    print("\n=== Testing SharedState ===")
    state = SharedState()
    
    # Initialize a task
    state.initialize_task("Test task", total_steps=3)
    print(f"Goal: {state.get('goal')}")
    print(f"Current step: {state.get_current_step()}")
    
    # Add some data
    state.set("project.name", "Test Project")
    state.add_file("test.py", "print('hello')")
    state.add_result(1, "code_agent", {"status": "success"})
    
    # Advance step
    new_step = state.advance_step()
    print(f"Advanced to step: {new_step}")
    
    # Get all state
    all_state = state.get_all()
    print(f"State keys: {list(all_state.keys())}")
    
    return state


def test_event_bus():
    """Test the EventBus component."""
    print("\n=== Testing EventBus ===")
    event_bus = get_global_event_bus()
    
    # Subscribe to events
    events_received = []
    
    async def event_handler(event):
        events_received.append(event.name)
        print(f"Received event: {event.name}")
    
    event_bus.subscribe("test_event", event_handler)
    
    # Emit events
    async def emit_events():
        await event_bus.emit("test_event", {"data": "test"})
        await event_bus.emit("test_event", {"data": "test2"})
    
    asyncio.run(emit_events())
    
    print(f"Events received: {events_received}")
    print(f"Subscribers: {event_bus.get_subscribers()}")
    
    return event_bus


def test_context_manager():
    """Test the ContextManager component."""
    print("\n=== Testing ContextManager ===")
    context_manager = ContextManager()
    
    # Build context for a task
    request = ContextRequest(
        task="Create a login page",
        agent="react",
        step=1,
        requirements={"memory": True, "files": False, "project": True}
    )
    
    context_bundle = context_manager.build_context(request)
    
    print(f"Task: {context_bundle.task}")
    print(f"Agent: {context_bundle.agent}")
    print(f"Context size: {context_bundle.get_total_size()} characters")
    print(f"Project context keys: {list(context_bundle.project_context.keys())}")
    
    return context_manager


def test_agent_registry():
    """Test the AgentRegistry component."""
    print("\n=== Testing AgentRegistry ===")
    registry = AgentRegistry()
    
    # Note: We can't register real agents without BaseAgent inheritance
    # This is a placeholder test
    
    print(f"Registered agents: {registry.get_agent_names()}")
    print(f"Agent count: {registry.get_count()}")
    
    return registry


def test_system_logger():
    """Test the SystemLogger component."""
    print("\n=== Testing SystemLogger ===")
    logger = get_global_logger()
    
    # Log some messages
    logger.info("TestComponent", "Test info message", {"key": "value"})
    logger.warning("TestComponent", "Test warning message")
    logger.error("TestComponent", "Test error message", {"error": "test error"})
    
    # Get logs
    logs = logger.get_logs(limit=5)
    print(f"Retrieved {len(logs)} logs")
    
    # Get stats
    stats = logger.get_log_stats()
    print(f"Total logs: {stats['total_logs']}")
    print(f"Level counts: {stats['level_counts']}")
    
    return logger


async def test_executor():
    """Test the Executor component."""
    print("\n=== Testing Executor ===")
    
    # Create components
    planner = Planner()
    executor = Executor(tool_manager=None, event_bus=get_global_event_bus())
    
    # Create a simple plan
    task_graph = planner.create_plan("Test task")
    
    # Note: Can't fully test without tool_manager and agent_registry
    # This is a placeholder test
    
    print(f"Plan created with {len(task_graph.steps)} steps")
    print("Executor requires tool_manager and agent_registry for full testing")
    
    return executor


def test_integration():
    """Test the integrated new architecture."""
    print("\n=== Testing Integration ===")
    
    # Initialize components
    shared_state = SharedState()
    event_bus = get_global_event_bus()
    logger = get_global_logger()
    planner = Planner()
    
    # Set up event handlers
    async def log_event(event):
        logger.info("EventBus", f"Event: {event.name}", event.data)
    
    event_bus.subscribe("test", log_event)
    
    # Simulate a workflow
    print("Simulating workflow:")
    
    # 1. Initialize task
    shared_state.initialize_task("Create login page", total_steps=5)
    logger.info("Integration", "Task initialized")
    
    # 2. Create plan
    task_graph = planner.create_plan("Create login page")
    logger.info("Integration", f"Plan created with {len(task_graph.steps)} steps")
    
    # 3. Emit event
    async def emit_test():
        await event_bus.emit("test", {"step": "plan_created"})
    
    asyncio.run(emit_test())
    
    # 4. Update state
    shared_state.set("current_phase", "planning")
    logger.info("Integration", "State updated")
    
    print(f"Current step: {shared_state.get_current_step()}")
    print(f"Current phase: {shared_state.get('current_phase')}")
    print(f"Plan steps: {len(task_graph.steps)}")
    
    print("Integration test completed successfully!")


if __name__ == "__main__":
    print("Testing New Architecture Components\n")
    
    try:
        # Test individual components
        test_planner()
        test_shared_state()
        test_event_bus()
        test_context_manager()
        test_agent_registry()
        test_system_logger()
        asyncio.run(test_executor())
        
        # Test integration
        test_integration()
        
        print("\n=== All Tests Completed Successfully ===")
        
    except Exception as e:
        print(f"\n=== Test Failed: {e} ===")
        import traceback
        traceback.print_exc()
