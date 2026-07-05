"""
Dagi-AI: An AI-powered development assistant with specialized agents.
"""

import os
import sys
from dotenv import load_dotenv
from agents import AgentManager
from tools.file_tool import FileTool
from tools.terminal_tool import TerminalTool
from tools.search_tool import SearchTool
from tools.git_tool import GitTool

# Load environment variables
load_dotenv()


def main():
    """Main entry point for Dagi-AI."""
    print("Dagi-AI: AI Development Assistant")
    print("=" * 50)
    
    # Initialize components
    agent_manager = AgentManager()
    file_tool = FileTool()
    terminal_tool = TerminalTool()
    search_tool = SearchTool()
    git_tool = GitTool()
    
    print("Agents initialized:")
    print(f"  - Code Agent: {agent_manager.code_agent.get_status()}")
    print(f"  - Memory Agent: {agent_manager.memory_agent.get_status()}")
    print(f"  - Docs Agent: {agent_manager.docs_agent.get_status()}")
    print(f"  - Planner Agent: {agent_manager.planner_agent.get_status()}")
    print()
    
    print("Tools initialized:")
    print("  - File Tool")
    print("  - Terminal Tool")
    print("  - Search Tool")
    print("  - Git Tool")
    print()
    
    # Example usage
    print("Example: Creating a plan for a task")
    plan_result = agent_manager.planner_agent.execute({
        'type': 'create_plan',
        'goal': 'Analyze and refactor codebase',
        'context': {'project_path': '.'}
    })
    print(f"Plan result: {plan_result}")
    print()
    
    print("Dagi-AI is ready. Use the API server (api/server.py) for full functionality.")
    print("To start the API server: python -m uvicorn api.server:app --reload")


if __name__ == "__main__":
    main()
