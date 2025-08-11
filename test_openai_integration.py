#!/usr/bin/env python3
"""
Test script to verify OpenAI GPT-OSS-20B integration
"""

import sys
import os
sys.path.append('backend')

from ai_generator import AIGenerator
from config import config
from search_tools import CourseSearchTool, ToolManager
from vector_store import VectorStore

# Fix the ChromaDB path for testing from root directory
config.CHROMA_PATH = "./backend/chroma_db"

def test_openai_integration():
    """Test the OpenAI integration"""
    print("Testing OpenAI GPT-OSS-20B integration...")
    print(f"Model: {config.OPENAI_MODEL}")
    print(f"Base URL: {config.OPENAI_BASE_URL}")
    print(f"API Key: {config.OPENAI_API_KEY[:10]}...")
    print(f"Timeout: {config.OPENAI_TIMEOUT} seconds")
    
    try:
        # Initialize components
        ai_generator = AIGenerator("openai")  # Use provider string instead of individual parameters
        
        # Test simple response without tools
        print("\n1. Testing simple response (no tools):")
        response = ai_generator.generate_response("Hello, how are you?")
        print(f"Response: {response}")
        
        # Test with tools
        print("\n2. Testing with search tools:")
        vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        search_tool = CourseSearchTool(vector_store)
        tool_manager = ToolManager()
        tool_manager.register_tool(search_tool)
        
        # Debug: Print tool definitions to see if they're correct
        tool_definitions = tool_manager.get_tool_definitions()
        print(f"Tool definitions: {tool_definitions}")
        
        # Test with MCP-specific question that should trigger search  
        print(f"Testing query: 'What is MCP and how do I use it in my applications?'")
        response_with_tools = ai_generator.generate_response(
            "What is MCP and how do I use it in my applications?",
            tools=tool_definitions,
            tool_manager=tool_manager
        )
        print(f"Response length: {len(response_with_tools)} characters")
        print(f"Response with tools: {response_with_tools[:500]}{'...' if len(response_with_tools) > 500 else ''}")
        
        # Check if any sources were found from the search
        sources = tool_manager.get_last_sources()
        source_links = tool_manager.get_last_source_links()
        print(f"Sources found: {sources}")
        print(f"Source links found: {len(source_links)} links")
        
        print("\n✅ OpenAI integration test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error testing OpenAI integration: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_openai_integration()
    sys.exit(0 if success else 1)