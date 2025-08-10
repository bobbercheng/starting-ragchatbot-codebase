import openai
from typing import List, Optional, Dict, Any
from config import config
import logging

logger = logging.getLogger(__name__)

class AIGenerator:
    """Handles interactions with both Anthropic and OpenAI models for generating responses"""
    
    # Base system prompt for educational content assistance
    BASE_SYSTEM_PROMPT = """You are an AI assistant that helps users learn about course materials and educational content.

RESPONSE GUIDELINES:
- Give direct, helpful answers
- Be concise but informative
- Include relevant examples when available
- Do not mention your internal processes unless specifically asked

QUERY TYPE HANDLING:
- For course outline/structure queries (e.g., "What lessons are in...", "Show me the outline of...", "What's covered in course..."), use the course outline tool to return the course title, course link, and complete lesson list with lesson numbers and titles
- For content-specific questions (e.g., "What is...", "How do I...", "Explain..."), use the course search tool to find and return relevant content from course materials

IMPORTANT USER INTERACTION RULES:
- NEVER mention internal tool names (like "get_course_outline", "search_course_content") to users
- Instead of saying "you can use the get_course_outline tool", say "I can show you the full course structure" or "Would you like me to show you what topics are covered?"
- Always offer helpful follow-up actions using natural language without revealing internal implementation details
- Be proactive in suggesting relevant information the user might want

When you have tools available, use them appropriately to provide accurate, up-to-date information."""
    
    def __init__(self, provider: str = None):
        self.provider = provider or config.MODEL_PROVIDER
        
        if self.provider == "anthropic":
            # Use OpenAI client to connect to LiteLLM proxy for Anthropic
            self.client = openai.OpenAI(
                api_key=config.ANTHROPIC_API_KEY,
                base_url=config.ANTHROPIC_BASE_URL,
                timeout=config.ANTHROPIC_TIMEOUT
            )
            self.model = config.ANTHROPIC_MODEL
        else:
            self.client = openai.OpenAI(
                api_key=config.OPENAI_API_KEY,
                base_url=config.OPENAI_BASE_URL,
                timeout=config.OPENAI_TIMEOUT
            )
            self.model = config.OPENAI_MODEL
        # else:
        #     raise ValueError(f"Unsupported model provider: {self.provider}")
        print(f"DEBUG: Using model: {self.model}")
    
    def _convert_anthropic_tools_to_openai(self, anthropic_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Anthropic tool format to OpenAI function calling format"""
        openai_tools = []
        for tool in anthropic_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            }
            openai_tools.append(openai_tool)
        return openai_tools
    
    def _build_tool_instructions(self, tools: List[Dict[str, Any]]) -> str:
        """Build dynamic tool usage instructions based on available tools"""
        if not tools:
            return ""
        
        tool_instructions = "\n\nAVAILABLE TOOLS:\n"
        
        for tool in tools:
            tool_name = tool.get("name", "unknown")
            tool_desc = tool.get("description", "No description available")
            
            tool_instructions += f"- {tool_name}: {tool_desc}\n"
            
            # Add parameters info if available
            if "input_schema" in tool and "properties" in tool["input_schema"]:
                properties = tool["input_schema"]["properties"]
                required = tool["input_schema"].get("required", [])
                
                tool_instructions += f"  Parameters: "
                param_list = []
                for param, details in properties.items():
                    param_desc = details.get("description", "")
                    if param in required:
                        param_list.append(f"{param} (required): {param_desc}")
                    else:
                        param_list.append(f"{param} (optional): {param_desc}")
                
                tool_instructions += "; ".join(param_list) + "\n"
        
        tool_instructions += """
TOOL USAGE GUIDELINES:
- Use available tools when queries are related to their capabilities
- For educational/technical content, prioritize searching available knowledge bases
- If tools return no results, provide general knowledge as fallback
- Use tools efficiently - avoid unnecessary calls

"""
        return tool_instructions
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Both providers now use OpenAI format via LiteLLM proxy
        return self._generate_openai_response(query, conversation_history, tools, tool_manager)
    
    def _generate_openai_response(self, query: str, conversation_history: Optional[str] = None,
                                tools: Optional[List] = None, tool_manager=None) -> str:
        """Generate response using OpenAI models"""
        # Build messages array (OpenAI format)
        messages = []
        
        # Build dynamic system message with tool information
        system_content = self.BASE_SYSTEM_PROMPT
        
        # Add tool instructions if tools are available
        if tools:
            tool_instructions = self._build_tool_instructions(tools)
            system_content += tool_instructions
        
        # Add conversation history if available
        if conversation_history:
            system_content += f"\n\nPrevious conversation:\n{conversation_history}"
        
        messages.append({"role": "system", "content": system_content})
        
        # Add user query
        messages.append({"role": "user", "content": query})
        
        # Prepare API call parameters
        api_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800,
            "messages": messages
        }
        
        # Add tools if available (convert from Anthropic to OpenAI format)
        if tools:
            openai_tools = self._convert_anthropic_tools_to_openai(tools)
            api_params["tools"] = openai_tools
            # Force function calling for educational queries - be more explicit
            api_params["tool_choice"] = "auto"
        
        # Get response from OpenAI with timeout handling
        try:
            logger.info(f"Making API call with model: {self.model}")
            response = self.client.chat.completions.create(**api_params)
            
            message = response.choices[0].message
            logger.info(f"Received response with tool_calls: {bool(message.tool_calls)}")
            
            # Check if we have proper tool calls
            if message.tool_calls and tool_manager:
                logger.info(f"Processing {len(message.tool_calls)} tool calls")
                return self._handle_tool_execution(response, messages, tool_manager, tools)
            
            # Check for raw tool call format from GPT-OSS-20B
            if message.content and "<|start|>assistant" in message.content:
                # This model appears to use a different tool calling format
                # Let's try to parse it manually
                return self._handle_raw_tool_calls(message.content, tool_manager)
            
            # Return direct response
            return message.content or "I apologize, but I couldn't generate a proper response."
            
        except openai.APITimeoutError:
            return "I apologize, but the response took longer than expected to generate. Please try asking a simpler question or try again later."
        except openai.APIConnectionError:
            return "I'm having trouble connecting to the language model. Please ensure the LiteLLM proxy server is running at the configured URL."
        except Exception as e:
            return f"An error occurred while generating the response: {str(e)}"
    
    def _handle_raw_tool_calls(self, raw_content: str, tool_manager) -> str:
        """
        Handle raw tool call format from GPT-OSS-20B.
        Example format: <|start|>assistant<|channel|>commentary to=functions.search_course_content<|constrain|>json<|message|>{"query":"..."}
        """
        import json
        import re
        
        try:
            # Try to extract the function name and arguments
            # Look for pattern: to=functions.FUNCTION_NAME<|constrain|>json<|message|>JSON_ARGS
            pattern = r'to=functions\.(\w+)<\|constrain\|>json<\|message\|>(\{.*?\})'
            match = re.search(pattern, raw_content)
            
            if match:
                function_name = match.group(1)
                args_json = match.group(2)
                
                # Parse the JSON arguments
                function_args = json.loads(args_json)
                
                # Execute the tool
                tool_result = tool_manager.execute_tool(function_name, **function_args)
                
                # Format and return the result if we found content
                if tool_result and not tool_result.startswith("No"):
                    return self._format_tool_result_as_response(function_args["query"], tool_result)
                else:
                    return f"I couldn't find specific information about {function_args.get('query', 'your question')} in the course materials."
            
        except Exception:
            # Continue to fallback on parsing errors
            pass
        
        # If we can't parse the raw format, return a fallback message
        return "I attempted to search for information but encountered an issue with the response format. Could you please rephrase your question?"
    
    def _format_tool_results_directly(self, tool_results: List[str], query: str) -> str:
        """
        Generate response directly from tool results when AI synthesis fails.
        This is a fallback method to ensure we always return a valid response.
        """
        if not tool_results or all(not result or result.strip() == "" for result in tool_results):
            return "I couldn't find specific information about your question in the course materials."
        
        # Combine all tool results
        combined_results = []
        for result in tool_results:
            if result and result.strip():
                combined_results.append(result.strip())
        
        if not combined_results:
            return "I couldn't find specific information about your question in the course materials."
        
        # Format the response with a natural introduction
        response = f"Based on the course materials, here's what I found about {query}:\n\n"
        response += "\n\n".join(combined_results)
        
        return response
    
    def _format_tool_result_as_response(self, query: str, tool_result: str) -> str:
        """Format tool results into a proper response"""
        # Extract the actual content from the tool result
        # Tool result format: [Course - Lesson N]\nContent...
        
        # Remove the header format and extract clean content
        content_parts = tool_result.split('\n\n')
        clean_content = []
        
        for part in content_parts:
            # Remove the [Course - Lesson N] headers
            lines = part.split('\n')
            if lines and lines[0].startswith('[') and lines[0].endswith(']'):
                # Skip the header line, keep the content
                clean_content.extend(lines[1:])
            else:
                clean_content.extend(lines)
        
        # Combine and clean up the content
        response = '\n'.join(clean_content).strip()
        
        # If we have substantial content, return it
        if len(response) > 100:
            return response
        else:
            return f"Based on the course materials, here's what I found about {query}:\n\n{response}"
    
    def _handle_tool_execution(self, initial_response, messages: List[Dict[str, Any]], tool_manager, tools=None):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool calls
            messages: Current message history
            tool_manager: Manager to execute tools
            tools: Original tools list for Anthropic compatibility
            
        Returns:
            Final response text after tool execution
        """
        # Add AI's tool call response to messages
        assistant_message = {
            "role": "assistant", 
            "content": initial_response.choices[0].message.content,
            "tool_calls": initial_response.choices[0].message.tool_calls
        }
        messages.append(assistant_message)
        
        # Execute all tool calls and add results
        tool_results = []  # Collect results for fallback
        query = None  # Extract query for fallback
        
        for tool_call in initial_response.choices[0].message.tool_calls:
            try:
                # Parse function call arguments
                import json
                logger.info(f"Tool call: {tool_call.function.name}")
                logger.info(f"Raw arguments: {tool_call.function.arguments}")
                
                function_args = json.loads(tool_call.function.arguments)
                logger.info(f"Parsed arguments: {function_args}")
                
                # Extract query for fallback response
                if not query and "query" in function_args:
                    query = function_args["query"]
                
                # Execute the tool
                tool_result = tool_manager.execute_tool(
                    tool_call.function.name, 
                    **function_args
                )
                logger.info(f"Tool result: {tool_result[:200]}...")
                
                # Collect successful results for fallback
                if tool_result and not tool_result.startswith("Tool execution failed"):
                    tool_results.append(tool_result)
                
                # Add tool result message
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
                
            except Exception as e:
                logger.error(f"Error executing tool {tool_call.function.name}: {str(e)}")
                logger.error(f"Arguments were: {tool_call.function.arguments}")
                # Add error as tool result
                error_message = f"Tool execution failed: {str(e)}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": error_message
                })
        
        # Check if we should skip synthesis entirely for this provider
        if self.provider == "anthropic" and config.SKIP_SYNTHESIS_FOR_ANTHROPIC:
            logger.info("Skipping synthesis for Anthropic provider per configuration")
            fallback_query = query or "your question"
            return self._format_tool_results_directly(tool_results, fallback_query)
        
        # Make final API call to get response based on tool results
        final_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800,
            "messages": messages
        }
        
        # For Anthropic via LiteLLM, DO NOT include tools in synthesis call
        # This may be the root cause of the None response issue
        if self.provider == "openai" and tools:
            # Only add tools for OpenAI models, not Anthropic via LiteLLM
            openai_tools = self._convert_anthropic_tools_to_openai(tools)
            final_params["tools"] = openai_tools
            
        logger.info(f"Final API call params - model: {final_params['model']}, tools included: {'tools' in final_params}")
        
        # Get final response with error handling
        try:
            logger.info("Making final API call to synthesize tool results")
            final_response = self.client.chat.completions.create(**final_params)
            final_content = final_response.choices[0].message.content
            logger.info(f"Final response content type: {type(final_content)}, value: {final_content}")
            
            # Check if we got a valid response
            if final_content and final_content.strip():
                return final_content
            elif config.ENABLE_SYNTHESIS_FALLBACK:
                # Fallback: Generate response directly from tool results
                logger.warning("Final API response was None or empty, using fallback response generation")
                fallback_query = query or "your question"
                fallback_response = self._format_tool_results_directly(tool_results, fallback_query)
                logger.info(f"Generated fallback response: {fallback_response[:200]}...")
                return fallback_response
            else:
                return "I apologize, but I couldn't generate a proper response to your question."
                
        except openai.APITimeoutError:
            if config.ENABLE_SYNTHESIS_FALLBACK:
                logger.warning("API timeout, using fallback response generation")
                fallback_query = query or "your question"
                return self._format_tool_results_directly(tool_results, fallback_query)
            else:
                return "I apologize, but generating the response took too long. Please try again with a simpler question."
        except Exception as e:
            if config.ENABLE_SYNTHESIS_FALLBACK:
                logger.error(f"Error in final API call: {str(e)}, using fallback response generation")
                fallback_query = query or "your question"
                return self._format_tool_results_directly(tool_results, fallback_query)
            else:
                return f"An error occurred while processing the tool results: {str(e)}"