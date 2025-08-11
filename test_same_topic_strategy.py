#!/usr/bin/env python3
"""
Test the updated same topic comparison strategy
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

def test_same_topic_strategy():
    """Test the updated same topic comparison strategy"""
    print("Testing Same Topic Comparison Strategy...")
    
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
        
        # Test the same topic comparison query that should use title-first approach
        print("\n=== Testing: 'Are there other courses that cover same topic as lesson 5 in MCP course?' ===")
        query = "Are there other courses that cover same topic as lesson 5 in MCP course?"
        
        response = ai_generator.generate_response(
            query,
            tools=tool_definitions,
            tool_manager=tool_manager
        )
        
        print(f"Response length: {len(response)} characters")
        print(f"Full response:")
        print("=" * 60)
        print(response)
        print("=" * 60)
        
        # Test another similar query
        print("\n\n=== Testing: 'What other lessons cover the same topic as Creating An MCP Client?' ===")
        query2 = "What other lessons cover the same topic as Creating An MCP Client?"
        
        tool_manager.reset_sources()
        response2 = ai_generator.generate_response(
            query2,
            tools=tool_definitions,
            tool_manager=tool_manager
        )
        
        print(f"Response length: {len(response2)} characters")
        print(f"Full response:")
        print("=" * 60)
        print(response2)
        print("=" * 60)
        
        print("\n✅ Same topic strategy test completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing same topic strategy: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_same_topic_strategy()
    sys.exit(0 if success else 1)