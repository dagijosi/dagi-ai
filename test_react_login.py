"""
Test the React login page task specifically to verify agents generate task-specific content.
"""

import asyncio
from agents.manager import ManagerAgent


async def test_react_login_task():
    """Test the React login page with form validation task."""
    print("=== Testing React Login Page Task ===\n")
    
    # Create manager
    manager = ManagerAgent()
    
    # Set up event handlers
    manager.setup_event_handlers()
    
    # Start the event bus
    await manager.event_bus.start()
    
    try:
        # Test the specific React login page task
        print("Testing: 'Write a React login page with form validation'")
        result = await manager.execute_with_new_architecture("Write a React login page with form validation")
        
        print(f"Goal: {result['goal']}")
        print(f"Plan steps: {len(result['plan']['steps'])}")
        print(f"Execution success: {result['execution']['success']}")
        print()
        
        # Show plan details
        print("Plan steps:")
        for step in result['plan']['steps']:
            print(f"  Step {step['step']}: {step['agent']} - {step['action']}")
            print(f"    Description: {step['description']}")
        print()
        
        # Show execution results for each step
        print("Execution results:")
        for step_result in result['execution']['results']:
            print(f"  Step {step_result['step']}: {step_result['success']}")
            if step_result['success']:
                result_data = step_result.get('result', {})
                if isinstance(result_data, dict):
                    if 'code' in result_data:
                        code = result_data['code']
                        print(f"    Code generated (first 200 chars): {code[:200]}...")
                        # Check if it's task-specific
                        if 'login' in code.lower() or 'form' in code.lower():
                            print(f"    ✅ Task-specific content detected!")
                        else:
                            print(f"    ❌ Generic content detected")
                    if 'design_spec' in result_data:
                        design = result_data['design_spec']
                        print(f"    Design spec (first 200 chars): {design[:200]}...")
                        # Check if it's task-specific
                        if 'login' in design.lower() or 'form' in design.lower():
                            print(f"    ✅ Task-specific design detected!")
                        else:
                            print(f"    ❌ Generic design detected")
            else:
                print(f"    Error: {step_result.get('error', 'Unknown error')}")
        print()
        
        # Check shared state
        print("=== Shared State ===")
        state = manager.shared_state.get_all()
        print(f"Goal: {state.get('goal')}")
        print(f"Files in state: {list(state.get('files', {}).keys())}")
        print()
        
    finally:
        await manager.event_bus.stop()
    
    print("=== Test Completed ===")


if __name__ == "__main__":
    asyncio.run(test_react_login_task())
