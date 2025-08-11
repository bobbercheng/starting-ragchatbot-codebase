#!/usr/bin/env python3
"""
Integration test for sequential tool calling functionality
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

def test_sequential_tool_calling():
    """Test sequential tool calling in a real scenario"""
    print("Testing Sequential Tool Calling Integration...")
    print(f"Sequential Tools Enabled: {config.ENABLE_SEQUENTIAL_TOOLS}")
    print(f"Max Rounds: {config.MAX_TOOL_ROUNDS}")
    
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
        
        # Test case that should trigger multiple rounds
        print("\n=== Testing Complex Query (should use multiple rounds) ===")
        complex_query = "What course teaches about MCP and what are the main topics covered in its first few lessons?"
        print(f"Query: {complex_query}")
        
        response = ai_generator.generate_response(
            complex_query,
            tools=tool_definitions,
            tool_manager=tool_manager
        )
        
        print(f"Response length: {len(response)} characters")
        print(f"Response preview: {response[:400]}...")
        
        sources = tool_manager.get_last_sources()
        print(f"Sources found: {len(sources)}")
        if sources:
            print(f"Sources: {sources[:3]}...")  # Show first 3
            
        # Test case that should be answered in one round
        print("\n=== Testing Simple Query (should use single round) ===")
        simple_query = "Hello, how are you?"
        print(f"Query: {simple_query}")
        
        tool_manager.reset_sources()  # Clear previous sources
        response2 = ai_generator.generate_response(
            simple_query,
            tools=tool_definitions,
            tool_manager=tool_manager
        )
        
        print(f"Response: {response2}")
        sources2 = tool_manager.get_last_sources()
        print(f"Sources used: {len(sources2)} (should be 0 for greeting)")
        
        # Test case with course outline + content search
        print("\n=== Testing Course Outline + Content Search ===")
        outline_query = "Show me the outline for the MCP course and tell me what lesson 3 covers specifically"
        print(f"Query: {outline_query}")
        
        tool_manager.reset_sources()
        response3 = ai_generator.generate_response(
            outline_query,
            tools=tool_definitions,
            tool_manager=tool_manager
        )
        
        print(f"Response length: {len(response3)} characters")
        print(f"Response preview: {response3[:400]}...")
        
        sources3 = tool_manager.get_last_sources()
        print(f"Sources used: {len(sources3)}")
        
        print("\n✅ Sequential tool calling integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing sequential tool calling: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sequential_tool_calling()
    sys.exit(0 if success else 1)