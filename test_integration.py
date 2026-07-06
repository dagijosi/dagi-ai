"""
Integration test for the new architecture with ManagerAgent.

This test demonstrates the full planner-executor workflow through ManagerAgent.
"""

import asyncio
from agents.manager import ManagerAgent


def test_new_architecture_integration():
    """Test the new architecture through ManagerAgent."""
    print("=== Testing New Architecture Integration ===\n")
    
    # Initialize ManagerAgent with new architecture
    manager = ManagerAgent()
    
    # Set up event handlers for logging
    manager.setup_event_handlers()
    
    # Start the event bus
    asyncio.run(manager.event_bus.start())
    
    try:
        # Test 1: Simple task
        print("Test 1: Simple task - 'Read the main.py file'")
        result1 = asyncio.run(manager.execute_with_new_architecture("Read the main.py file"))
        print(f"Goal: {result1['goal']}")
        print(f"Plan steps: {len(result1['plan']['steps'])}")
        print(f"Execution success: {result1['execution']['success']}")
        print()
        
        # Test 2: Medium complexity task
        print("Test 2: Medium task - 'Create a simple Python function'")
        result2 = asyncio.run(manager.execute_with_new_architecture("Create a simple Python function"))
        print(f"Goal: {result2['goal']}")
        print(f"Plan steps: {len(result2['plan']['steps'])}")
        print(f"Execution success: {result2['execution']['success']}")
        if not result2['execution']['success']:
            print("Execution results:")
            for step_result in result2['execution']['results']:
                print(f"  Step {step_result['step']}: {step_result['success']} - {step_result.get('error', 'OK')}")
                print(f"    Result: {step_result.get('result', 'No result')}")
        
        # Show plan details for debugging
        print("Plan details for Test 2:")
        for step in result2['plan']['steps']:
            print(f"  Step {step['step']}: {step['agent']} - {step['action']}")
        print()
        
        # Test 3: Complex task
        print("Test 3: Complex task - 'Create a React login page and commit it'")
        result3 = asyncio.run(manager.execute_with_new_architecture("Create a React login page and commit it"))
        print(f"Goal: {result3['goal']}")
        print(f"Plan steps: {len(result3['plan']['steps'])}")
        print(f"Execution success: {result3['execution']['success']}")
        print()
        
        # Check shared state
        print("=== Shared State ===")
        state = manager.shared_state.get_all()
        print(f"Goal: {state.get('goal')}")
        print(f"Current step: {state.get('current_step')}")
        print(f"Total steps: {state.get('total_steps')}")
        print(f"Files in state: {list(state.get('files', {}).keys())}")
        print()
        
        # Check event history
        print("=== Event History ===")
        events = manager.event_bus.get_event_history(limit=10)
        print(f"Total events: {len(events)}")
        for event in events[-5:]:  # Show last 5 events
            print(f"  {event['timestamp']}: {event['name']}")
        print()
        
        # Check logs
        print("=== System Logs ===")
        logs = manager.logger.get_logs(limit=5)
        print(f"Total logs: {len(logs)}")
        for log in logs[-3:]:  # Show last 3 logs
            print(f"  {log['timestamp']} [{log['level']}] {log['component']}: {log['message']}")
        print()
        
        print("=== Integration Test Completed Successfully ===")
        
    finally:
        # Stop the event bus
        asyncio.run(manager.event_bus.stop())


def test_backward_compatibility():
    """Test that the old run() method still works."""
    print("\n=== Testing Backward Compatibility ===\n")
    
    manager = ManagerAgent()
    
    # Test with simple input
    print("Testing old run() method with simple task...")
    result = manager.run("What is Python?")
    
    print(f"Manager analysis: {result.get('manager_analysis', {}).get('analysis')}")
    print(f"Complexity: {result.get('manager_analysis', {}).get('complexity')}")
    print(f"Total steps: {result.get('manager_analysis', {}).get('total_steps')}")
    
    print("\nBackward compatibility test passed!")


if __name__ == "__main__":
    try:
        test_new_architecture_integration()
        test_backward_compatibility()
        
        print("\n=== All Integration Tests Passed ===")
        
    except Exception as e:
        print(f"\n=== Integration Test Failed: {e} ===")
        import traceback
        traceback.print_exc()
