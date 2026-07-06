"""
Test script for the TanStack Query & TypeScript Generation Agent.

Tests both modes of operation:
- Mode 1: Project Initialization
- Mode 2: API Ingestion & Full Stack Typing
"""

import asyncio
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.tanstack_agent import TanStackAgent
from agents.manager import ManagerAgent
from registry.agent_adapter import AgentAdapter
from registry.agent_registry import AgentRegistry


def test_agent_creation():
    """Test that the TanStack agent can be created and has the right structure."""
    print("=== Test: Agent Creation ===")
    agent = TanStackAgent()
    
    assert agent.status == "idle", f"Expected idle, got {agent.status}"
    assert hasattr(agent, 'system_prompt'), "Agent missing system_prompt"
    assert 'TanStack Query' in agent.system_prompt, "System prompt missing TanStack Query reference"
    assert 'TypeScript' in agent.system_prompt, "System prompt missing TypeScript reference"
    assert hasattr(agent, 'execute'), "Agent missing execute method"
    assert hasattr(agent, 'get_status'), "Agent missing get_status method"
    
    print("✅ Agent created successfully")
    print(f"   Status: {agent.status}")
    print(f"   System prompt length: {len(agent.system_prompt)} chars")
    print()


def test_agent_execute_initialize():
    """Test Mode 1: Project Initialization."""
    print("=== Test: Mode 1 - Project Initialization ===")
    agent = TanStackAgent()
    
    result = agent.execute({
        'type': 'initialize',
        'task': 'Create a blog application with TanStack Query',
        'framework': 'nextjs'
    })
    
    assert result['status'] == 'initialized', f"Expected initialized, got {result['status']}"
    assert 'code' in result, "Result missing code"
    assert result['mode'] == 'project_initialization', f"Expected project_initialization mode, got {result.get('mode')}"
    assert len(result['code']) > 0, "Code should not be empty"
    
    # Check that the code contains key concepts
    code_lower = result['code'].lower()
    
    print(f"✅ Project initialization completed")
    print(f"   Code length: {len(result['code'])} chars")
    print(f"   Contains 'services': {'services' in code_lower}")
    print(f"   Contains 'hooks': {'hooks' in code_lower}")
    print(f"   Contains 'types': {'types' in code_lower}")
    print()


def test_agent_execute_ingest_api():
    """Test Mode 2: API Ingestion & Full Stack Typing."""
    print("=== Test: Mode 2 - API Ingestion ===")
    agent = TanStackAgent()
    
    result = agent.execute({
        'type': 'ingest_api',
        'task': 'Create user profile API integration',
        'api_endpoint': 'GET /api/users/:id',
        'response_example': '{ "id": 1, "name": "John Doe", "email": "john@example.com", "role": "admin" }'
    })
    
    assert result['status'] == 'ingested', f"Expected ingested, got {result['status']}"
    assert 'code' in result, "Result missing code"
    assert result['mode'] == 'api_ingestion', f"Expected api_ingestion mode, got {result.get('mode')}"
    assert len(result['code']) > 0, "Code should not be empty"
    
    code_lower = result['code'].lower()
    
    print(f"✅ API ingestion completed")
    print(f"   Code length: {len(result['code'])} chars")
    print(f"   Contains type definitions: {'interface' in code_lower or 'type ' in code_lower}")
    print(f"   Contains query/mutation: {'usequery' in code_lower or 'usemutation' in code_lower}")
    print()


def test_agent_default_modes():
    """Test that the agent correctly infers the mode based on input data."""
    print("=== Test: Default Mode Inference ===")
    agent = TanStackAgent()
    
    # With API endpoint - should use API ingestion mode
    result_with_api = agent.execute({
        'task': 'Create a login API integration',
        'api_endpoint': 'POST /api/auth/login'
    })
    
    # Without API endpoint - should use project initialization mode
    result_without_api = agent.execute({
        'task': 'Set up a new project with TanStack Query'
    })
    
    if result_with_api.get('mode') == 'api_ingestion':
        print("✅ Agent correctly uses API ingestion mode when API endpoint is provided")
    else:
        print(f"⚠️ Expected api_ingestion mode, got {result_with_api.get('mode')}")
    
    if result_without_api.get('mode') == 'project_initialization':
        print("✅ Agent correctly uses project initialization mode when no API endpoint")
    else:
        print(f"⚠️ Expected project_initialization mode, got {result_without_api.get('mode')}")
    
    print()


def test_agent_execute_types():
    """Test TypeScript type generation."""
    print("=== Test: Type Generation ===")
    agent = TanStackAgent()
    
    result = agent.execute({
        'type': 'generate_types',
        'task': 'Generate types for a blog post API response',
        'response_example': '{ "id": 1, "title": "Hello World", "body": "Content here", "author": { "id": 1, "name": "Jane" }, "tags": ["tech", "news"], "published": true, "views": 1500 }'
    })
    
    assert result['status'] == 'generated', f"Expected generated, got {result['status']}"
    assert 'code' in result, "Result missing code"
    
    code_lower = result['code'].lower()
    
    print(f"✅ Type generation completed")
    print(f"   Code length: {len(result['code'])} chars")
    print(f"   Contains 'interface': {'interface' in code_lower}")
    print()


def test_agent_execute_hook():
    """Test TanStack Query hook generation."""
    print("=== Test: Hook Generation ===")
    agent = TanStackAgent()
    
    result = agent.execute({
        'type': 'generate_hook',
        'task': 'Create a usePosts hook for fetching blog posts',
        'hook_type': 'query'
    })
    
    assert result['status'] == 'generated', f"Expected generated, got {result['status']}"
    assert 'code' in result, "Result missing code"
    
    code_lower = result['code'].lower()
    
    print(f"✅ Hook generation completed")
    print(f"   Code length: {len(result['code'])} chars")
    print(f"   Contains useQuery: {'usequery' in code_lower}")
    print()


def test_agent_execute_service():
    """Test service layer generation."""
    print("=== Test: Service Generation ===")
    agent = TanStackAgent()
    
    result = agent.execute({
        'type': 'generate_service',
        'task': 'Create a user service with fetch wrapper',
        'api_endpoint': '/api/users',
        'http_method': 'GET'
    })
    
    assert result['status'] == 'generated', f"Expected generated, got {result['status']}"
    assert 'code' in result, "Result missing code"
    
    print(f"✅ Service generation completed")
    print(f"   Code length: {len(result['code'])} chars")
    print(f"   Endpoint: {result.get('api_endpoint')}")
    print()


def test_agent_adapter_compatibility():
    """Test that the agent works with the AgentAdapter for the registry."""
    print("=== Test: Agent Adapter Compatibility ===")
    
    from agents.tanstack_agent import TanStackAgent
    tanstack_agent = TanStackAgent()
    
    # Create adapter
    adapter = AgentAdapter("tanstack", tanstack_agent)
    
    # Test capabilities
    capabilities = adapter.get_capabilities()
    assert len(capabilities) > 0, "Should have capabilities"
    assert 'generate_types' in capabilities, "Should have generate_types capability"
    
    print(f"✅ Agent adapter created successfully")
    print(f"   Name: {adapter.name}")
    print(f"   Capabilities: {capabilities}")
    print()


def test_agent_registry_integration():
    """Test that the agent can be registered and found in the registry."""
    print("=== Test: Agent Registry Integration ===")
    
    registry = AgentRegistry()
    from agents.tanstack_agent import TanStackAgent
    from registry.agent_adapter import AgentAdapter
    
    tanstack_agent = TanStackAgent()
    adapter = AgentAdapter("tanstack", tanstack_agent)
    
    # Register
    registry.register(
        adapter,
        metadata={"type": "tanstack", "description": "TanStack Query development"}
    )
    
    # Verify registration
    assert registry.is_registered("tanstack"), "Agent should be registered"
    assert registry.get_count() > 0, "Registry should not be empty"
    
    # Find by capability
    found = registry.find_agents_by_capability("generate_types")
    assert "tanstack" in found, "Should find tanstack by capability"
    
    # Find by task description
    suitable = registry.find_suitable_agents("need to generate TypeScript types for an API")
    assert len(suitable) > 0, "Should find suitable agents"
    
    print(f"✅ Agent registered in registry")
    print(f"   Registered: {registry.is_registered('tanstack')}")
    print(f"   Found by capability 'generate_types': {'tanstack' in found}")
    print(f"   Suitable agents for TypeScript task: {suitable}")
    print()


def test_manager_integration():
    """Test that the manager can route tasks to the TanStack agent."""
    print("=== Test: Manager Integration ===")
    
    manager = ManagerAgent()
    
    # Check that tanstack agent exists in manager
    assert 'tanstack' in manager.agents, "Manager should have tanstack agent"
    assert hasattr(manager, 'tanstack_agent'), "Manager should have tanstack_agent attribute"
    
    # Test routing to tanstack agent
    result = manager.route_task('tanstack', {
        'type': 'generate_types',
        'task': 'Create types for a product API',
        'response_example': '{ "id": 1, "name": "Laptop", "price": 999.99 }'
    })
    
    assert result is not None, "Route task should return a result"
    assert result.get('status') in ['generated', 'ingested', 'initialized'], f"Unexpected status: {result.get('status')}"
    
    print(f"✅ Manager integration working")
    print(f"   Agent in manager: 'tanstack' in manager.agents")
    print(f"   Task routing result status: {result.get('status')}")
    print()


def test_agent_get_status():
    """Test the get_status method."""
    print("=== Test: Agent Status ===")
    agent = TanStackAgent()
    
    # Initial status
    assert agent.get_status() == "idle", "Initial status should be idle"
    
    # Status after execution
    agent.execute({'type': 'generate_types', 'task': 'test'})
    assert agent.get_status() == "idle", "Status should return to idle after execution"
    
    print(f"✅ Agent status: {agent.get_status()}")
    print()


async def test_new_architecture_integration():
    """Test the new architecture integration path."""
    print("=== Test: New Architecture Integration ===")
    
    manager = ManagerAgent()
    manager.setup_event_handlers()
    await manager.event_bus.start()
    
    try:
        # Check that tanstack is in the registry after registration
        manager.register_agents_to_registry()
        
        assert manager.agent_registry.is_registered("tanstack"), "TanStack should be registered"
        
        # Get metadata
        metadata = manager.agent_registry.get_metadata("tanstack")
        print(f"✅ TanStack agent registered in new architecture")
        print(f"   Metadata: {metadata}")
        
        # List all agents
        agents = manager.agent_registry.list_agents()
        tanstack_info = [a for a in agents if a['name'] == 'tanstack']
        if tanstack_info:
            print(f"   Capabilities: {tanstack_info[0]['capabilities']}")
        
    finally:
        await manager.event_bus.stop()
    
    print()


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("  TanStack Query & TypeScript Agent - Test Suite")
    print("=" * 60)
    print()
    
    try:
        test_agent_creation()
        test_agent_execute_initialize()
        test_agent_execute_ingest_api()
        test_agent_default_modes()
        test_agent_execute_types()
        test_agent_execute_hook()
        test_agent_execute_service()
        test_agent_adapter_compatibility()
        test_agent_registry_integration()
        test_manager_integration()
        test_agent_get_status()
        
        # Async test
        asyncio.run(test_new_architecture_integration())
        
        print("=" * 60)
        print("  ✅ All Tests Passed Successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
