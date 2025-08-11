#!/usr/bin/env python3
import sys
sys.path.append('backend')

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import VectorStore
from config import config

config.CHROMA_PATH = './backend/chroma_db'
vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)

# Test Round 1 result
search_tool = CourseSearchTool(vector_store)
result1 = search_tool.execute(query='lesson 5', course_name='MCP', lesson_number=5)

print("=== Round 1 Result Preview ===")
print(result1[:500])

print("\n=== Searching for Lesson Title Patterns ===")
patterns = ['Building the MCP Client', 'Creating an MCP Client', 'MCP Client', 'Building MCP Client']
for pattern in patterns:
    print(f"'{pattern}' found: {pattern in result1}")

print("\n=== Testing Round 2 Query ===")
# Test what Round 2 should search for
result2 = search_tool.execute(query='Building the MCP Client')
print(f"Round 2 result length: {len(result2)}")
print("Round 2 preview:", result2[:300])