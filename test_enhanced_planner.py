"""
Test the enhanced planner with all agent types.
"""

from planner.planner import Planner


def test_all_agent_types():
    """Test that the planner utilizes all registered agents."""
    print("=== Testing Enhanced Planner with All Agent Types ===\n")
    
    planner = Planner()
    
    test_cases = [
        {
            "name": "React Login Page",
            "input": "Create a React login page with form validation",
            "expected_agents": ["search", "ui_designer", "react", "code", "file", "memory"]
        },
        {
            "name": "Flutter App",
            "input": "Build a Flutter mobile app with authentication",
            "expected_agents": ["search", "ui_designer", "flutter", "code", "file", "memory"]
        },
        {
            "name": "API Design",
            "input": "Design a REST API for user management",
            "expected_agents": ["search", "api", "memory"]
        },
        {
            "name": "Database Schema",
            "input": "Design a database schema for an e-commerce system",
            "expected_agents": ["search", "database_expert", "memory"]
        },
        {
            "name": "Security Audit",
            "input": "Perform security audit on authentication system",
            "expected_agents": ["security_auditor", "memory"]
        },
        {
            "name": "Full Stack with Testing",
            "input": "Create a full-stack app with tests and deploy it",
            "expected_agents": ["search", "code", "file", "qa_tester", "devops", "memory"]
        },
        {
            "name": "Simple Code",
            "input": "Create a simple Python function",
            "expected_agents": ["search", "code", "file", "memory"]
        }
    ]
    
    for test_case in test_cases:
        print(f"Test: {test_case['name']}")
        print(f"Input: {test_case['input']}")
        
        task_graph = planner.create_plan(test_case['input'])
        
        agents_used = [step.agent for step in task_graph.steps]
        print(f"Agents used: {agents_used}")
        print(f"Total steps: {len(task_graph.steps)}")
        
        # Check if expected agents are present
        missing_agents = set(test_case['expected_agents']) - set(agents_used)
        if missing_agents:
            print(f"⚠️  Missing expected agents: {missing_agents}")
        else:
            print("✅ All expected agents present")
        
        # Show plan details
        print("Plan steps:")
        for step in task_graph.steps:
            print(f"  Step {step.step}: {step.agent} - {step.action}")
        
        print()
    
    print("=== Enhanced Planner Test Completed ===")


if __name__ == "__main__":
    test_all_agent_types()
