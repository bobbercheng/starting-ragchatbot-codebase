#!/usr/bin/env python3
"""
Test script to verify enhanced logging for sequential tool calling
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

def test_enhanced_logging():
    """Test the enhanced logging with queries that should show different behaviors"""
    print("Testing Enhanced Logging for Sequential Tool Calling...")
    
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
        
        # Test the problematic query that should show Round 2 behavior
        print("\n=== Testing: 'What was covered in lesson 5 of the MCP course?' ===")
        query = "What was covered in lesson 5 of the MCP course?"
        
        response = ai_generator.generate_response(
            query,
            tools=tool_definitions,
            tool_manager=tool_manager
        )
        
        print(f"Response length: {len(response)} characters")
        print(f"Response preview: {response[:300]}...")
        
        # Test a simple greeting that should not use tools
        print("\n\n=== Testing: 'Hello, how are you?' ===")
        query2 = "Hello, how are you?"
        
        tool_manager.reset_sources()
        response2 = ai_generator.generate_response(
            query2,
            tools=tool_definitions,
            tool_manager=tool_manager
        )
        
        print(f"Response: {response2}")
        
        print("\n✅ Enhanced logging test completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing enhanced logging: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_logging()
    sys.exit(0 if success else 1)