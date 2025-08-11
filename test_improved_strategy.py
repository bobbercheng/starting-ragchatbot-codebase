#!/usr/bin/env python3
"""
Test the improved sequential tool calling strategy
"""

import sys
import os
sys.path.append('backend')

from ai_generator import AIGenerator
from config import config
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import VectorStore

# Fix the ChromaDB path for testing from root directory
config.CHROMA_PATH = "./backend/chroma_db"

def test_improved_strategy():
    """Test the improved strategy with lesson-specific queries"""
    print("Testing Improved Sequential Tool Calling Strategy...")
    
    try:
        # Initialize components
        ai_generator = AIGenerator("openai")
        vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        
        # Set up tools
        search_tool = CourseSearchTool(vector_store)
        outline_tool = CourseOutlineTool(vector_store)
        tool_manager = ToolManager()
        tool_manager.register_tool(search_tool)
        tool_manager.register_tool(outline_tool)
        
        tool_definitions = tool_manager.get_tool_definitions()
        print(f"Available tools: {[tool['name'] for tool in tool_definitions]}")
        
        # Test with a working query first to check database
        print("\n=== Testing: 'What is MCP and how do I use it in my applications?' ===")
        response1 = ai_generator.generate_response(
            "What is MCP and how do I use it in my applications?",
            tools=tool_definitions,
            tool_manager=tool_manager
        )
        print(f"Response length: {len(response1)}")
        print(f"Success: {len(response1) > 500}")
        
        # Now test lesson-specific query
        print("\n=== Testing: 'What was covered in lesson 5 of the MCP course?' ===")
        tool_manager.reset_sources()
        response2 = ai_generator.generate_response(
            "What was covered in lesson 5 of the MCP course?",
            tools=tool_definitions,
            tool_manager=tool_manager
        )
        print(f"Response length: {len(response2)}")
        print(f"Sources found: {len(tool_manager.get_last_sources())}")
        
        # Test another lesson query to see strategy
        print("\n=== Testing: 'What topics are covered in lesson 1?' ===")
        tool_manager.reset_sources()
        response3 = ai_generator.generate_response(
            "What topics are covered in lesson 1?",
            tools=tool_definitions,
            tool_manager=tool_manager
        )
        print(f"Response length: {len(response3)}")
        
        print("\n✅ Strategy test completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing strategy: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_improved_strategy()
    sys.exit(0 if success else 1)