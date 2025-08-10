from typing import Dict, Any, Optional, Protocol, List
from abc import ABC, abstractmethod
import json
from vector_store import VectorStore, SearchResults


class Tool(ABC):
    """Abstract base class for all tools"""
    
    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search
        self.last_source_links = []  # Track source links from last search
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "What to search for in the course content"
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)"
                    }
                },
                "required": ["query"]
            }
        }
    
    def execute(self, query: str, course_name: Optional[str] = None, lesson_number: Optional[int] = None) -> str:
        """
        Execute the search tool with given parameters.
        
        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter
            
        Returns:
            Formatted search results or error message
        """
        
        # Use the vector store's unified search interface
        results = self.store.search(
            query=query,
            course_name=course_name,
            lesson_number=lesson_number
        )
        
        # Handle errors
        if results.error:
            return results.error
        
        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."
        
        # Format and return results
        return self._format_results(results)
    
    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        unique_sources = {}  # Track unique sources for the UI (key: course+lesson, value: source info)
        
        # Get unique courses to minimize database queries
        unique_courses = set()
        for meta in results.metadata:
            course_title = meta.get('course_title')
            if course_title:
                unique_courses.add(course_title)
        
        # Retrieve lesson metadata for all unique courses
        course_lessons_map = self._get_course_lessons_map(list(unique_courses))
        
        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get('course_title', 'unknown')
            lesson_num = meta.get('lesson_number')
            
            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"
            
            # Create unique key for deduplication
            source_key = f"{course_title}|{lesson_num}"
            
            # Only add source if we haven't seen this course+lesson combination
            if source_key not in unique_sources:
                # Build source text
                source = course_title
                if lesson_num is not None:
                    source += f" - Lesson {lesson_num}"
                
                # Find the lesson link
                lesson_link = None
                if course_title in course_lessons_map and lesson_num is not None:
                    lessons = course_lessons_map[course_title]
                    for lesson in lessons:
                        if lesson.get('lesson_number') == lesson_num:
                            lesson_link = lesson.get('lesson_link')
                            break
                
                # Store unique source info
                unique_sources[source_key] = {
                    'text': source,
                    'url': lesson_link
                }
            
            formatted.append(f"{header}\n{doc}")
        
        # Convert unique sources to lists for storage
        self.last_sources = [info['text'] for info in unique_sources.values()]
        self.last_source_links = [info for info in unique_sources.values()]
        
        return "\n\n".join(formatted)
    
    def _get_course_lessons_map(self, course_titles: List[str]) -> Dict[str, List[Dict]]:
        """Retrieve lesson metadata for multiple courses from the catalog"""
        course_lessons_map = {}
        
        for course_title in course_titles:
            try:
                # Query the course catalog for this specific course
                results = self.store.course_catalog.get(
                    ids=[course_title]
                )
                
                if results and 'metadatas' in results and results['metadatas']:
                    metadata = results['metadatas'][0]
                    lessons_json = metadata.get('lessons_json')
                    
                    if lessons_json:
                        try:
                            lessons = json.loads(lessons_json)
                            course_lessons_map[course_title] = lessons
                        except json.JSONDecodeError:
                            print(f"Error parsing lessons JSON for course: {course_title}")
                            course_lessons_map[course_title] = []
                    else:
                        course_lessons_map[course_title] = []
                        
            except Exception as e:
                print(f"Error retrieving lessons for course {course_title}: {e}")
                course_lessons_map[course_title] = []
        
        return course_lessons_map


class CourseOutlineTool(Tool):
    """Tool for getting course outline information (title, link, and lesson list)"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "get_course_outline",
            "description": "Get course outline including title, course link, and complete lesson list",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_title": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    }
                },
                "required": ["course_title"]
            }
        }
    
    def execute(self, course_title: str) -> str:
        """
        Execute the course outline tool to get course metadata and lesson list.
        
        Args:
            course_title: Course title to get outline for
            
        Returns:
            Formatted course outline or error message
        """
        # Resolve course name using semantic search
        resolved_course_title = self.store._resolve_course_name(course_title)
        
        if not resolved_course_title:
            return f"No course found matching '{course_title}'. Please check the course name and try again."
        
        # Get course metadata from the catalog
        try:
            results = self.store.course_catalog.get(ids=[resolved_course_title])
            
            if not results or not results.get('metadatas') or not results['metadatas']:
                return f"No course details found for '{resolved_course_title}'"
            
            metadata = results['metadatas'][0]
            
            # Extract course information
            course_title_actual = metadata.get('title', resolved_course_title)
            course_link = metadata.get('course_link')
            instructor = metadata.get('instructor')
            lessons_json = metadata.get('lessons_json')
            
            # Format the response
            response_parts = []
            response_parts.append(f"Course Title: {course_title_actual}")
            
            if course_link:
                response_parts.append(f"Course Link: {course_link}")
            
            if instructor:
                response_parts.append(f"Course Instructor: {instructor}")
            
            # Parse and display lessons
            if lessons_json:
                try:
                    lessons = json.loads(lessons_json)
                    if lessons:
                        response_parts.append("\nLessons:")
                        for lesson in lessons:
                            lesson_num = lesson.get('lesson_number')
                            lesson_title = lesson.get('lesson_title')
                            lesson_link = lesson.get('lesson_link')
                            
                            lesson_line = f"  Lesson {lesson_num}: {lesson_title}"
                            if lesson_link:
                                lesson_line += f" (Link: {lesson_link})"
                            response_parts.append(lesson_line)
                    else:
                        response_parts.append("\nNo lessons found for this course.")
                except json.JSONDecodeError:
                    response_parts.append("\nError: Could not parse lesson information.")
            else:
                response_parts.append("\nNo lesson information available for this course.")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            return f"Error retrieving course outline for '{course_title}': {str(e)}"


class ToolManager:
    """Manages available tools for the AI"""
    
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    
    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"
        
        return self.tools[tool_name].execute(**kwargs)
    
    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources') and tool.last_sources:
                return tool.last_sources
        return []
    
    def get_last_source_links(self) -> list:
        """Get source links from the last search operation"""
        # Check all tools for last_source_links attribute
        for tool in self.tools.values():
            if hasattr(tool, 'last_source_links') and tool.last_source_links:
                return tool.last_source_links
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources'):
                tool.last_sources = []
            if hasattr(tool, 'last_source_links'):
                tool.last_source_links = []