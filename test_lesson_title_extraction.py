#!/usr/bin/env python3
"""
Test lesson title extraction from course outline
"""

import sys
import os
sys.path.append('backend')

from search_tools import CourseOutlineTool, CourseSearchTool
from vector_store import VectorStore
from config import config

# Fix the ChromaDB path for testing from root directory
config.CHROMA_PATH = "./backend/chroma_db"

def test_lesson_title_extraction():
    """Test exact lesson title extraction from course outline"""
    print("Testing Lesson Title Extraction...")
    
    try:
        vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        
        # Test course outline tool
        outline_tool = CourseOutlineTool(vector_store)
        outline_result = outline_tool.execute("MCP")
        
        print("=== Course Outline Result ===")
        print(outline_result)
        
        print("\n=== Looking for Lesson 5 Title ===")
        lines = outline_result.split('\n')
        lesson_5_title = None
        
        for line in lines:
            if 'Lesson 5:' in line:
                print(f"Found line: {line}")
                # Extract just the title part
                if ':' in line:
                    lesson_5_title = line.split(':', 1)[1].strip()
                    if '(' in lesson_5_title:
                        lesson_5_title = lesson_5_title.split('(')[0].strip()
                    print(f"Extracted title: '{lesson_5_title}'")
                break
        
        if lesson_5_title:
            print(f"\n=== Testing Search with Exact Title: '{lesson_5_title}' ===")
            search_tool = CourseSearchTool(vector_store)
            search_result = search_tool.execute(query=lesson_5_title)
            
            print(f"Search result length: {len(search_result)}")
            print(f"Search result preview: {search_result[:300]}...")
        else:
            print("Could not find Lesson 5 title in outline")
        
        print("\n✅ Lesson title extraction test completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing lesson title extraction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_lesson_title_extraction()
    sys.exit(0 if success else 1)