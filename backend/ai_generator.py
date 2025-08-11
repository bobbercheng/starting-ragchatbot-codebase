import openai
from typing import List, Optional, Dict, Any
from config import config
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ToolRoundResult:
    """Result from executing one round of tool calls"""
    round_number: int
    had_tool_calls: bool
    tool_results: List[str]
    queries_executed: List[str]
    should_stop: bool
    error: Optional[str] = None

class AIGenerator:
    """Handles interactions with both Anthropic and OpenAI models for generating responses"""
    
    # Base system prompt for educational content assistance
    BASE_SYSTEM_PROMPT = """You are an AI assistant that helps users learn about course materials and educational content.

== RESPONSE GUIDELINES ==
- Give direct, helpful answers
- Be concise but informative
- Include relevant examples when available
- Do not mention your internal processes unless specifically asked

== QUERY TYPE HANDLING ==
- For course outline/structure queries (e.g., "What lessons are in...", "Show me the outline of...", "What's covered in course..."), use the course outline tool to return the course title, course link, and complete lesson list with lesson numbers and titles
- For content-specific questions (e.g., "What is...", "How do I...", "Explain..."), use the course search tool to find and return relevant content from course materials

== USER INTERACTION RULES ==
- NEVER mention internal tool names (like "get_course_outline", "search_course_content") to users
- Instead of saying "you can use the get_course_outline tool", say "I can show you the full course structure" or "Would you like me to show you what topics are covered?"
- Always offer helpful follow-up actions using natural language without revealing internal implementation details
- Be proactive in suggesting relevant information the user might want

== TOOL USAGE POLICY ==
When you have tools available, use them appropriately to provide accurate, up-to-date information.

== MULTI-STEP REASONING FRAMEWORK ==
You can make multiple tool calls across up to 2 rounds to gather comprehensive information:
- Round 1: Make initial searches based on the user's question
- Round 2: If needed, refine your search based on initial results or search for additional related information
- After each round, you'll see the results and can decide if more information is needed

Use follow-up searches to:
  * Get more specific information if initial search was too broad
  * Search different courses/lessons if initial search found limited results
  * Look for related concepts or examples to provide comprehensive coverage
- Always aim to provide complete, helpful answers using all available information

== OPTIMAL SEARCH STRATEGY ==
Content-First Approach - Choose strategy based on query type:

**For Lesson-Specific Queries** (e.g., "What was covered in lesson 5?"):
1. Round 1: DISCOVERY PHASE
   - Use course search tool with course_name and lesson_number to find specific content
   - Goal: Find if course exists and get lesson content
   - Benefit: Direct answer to user's question + course discovery

2. Round 2: CONTEXT ENRICHMENT PHASE
   - Use course outline tool with exact course title from Round 1
   - Goal: Add course structure context to enrich the response
   - Benefit: Full context for better synthesis

**For "Same Topic" Comparison Queries** (e.g., "Are there other courses that cover same topic as lesson X?"):
1. Round 1: TITLE DISCOVERY PHASE
   - Use course outline tool to get the exact lesson title (e.g., "Creating An MCP Client")
   - Goal: Get the precise lesson title without LLM modification
   - CRITICAL: Use the EXACT lesson title from the course outline, do not paraphrase or change it

2. Round 2: TOPIC SEARCH PHASE
   - Use course search tool with query="{exact_lesson_title}" (NO course filter)
   - Goal: Search across ALL courses for content matching the exact lesson title
   - Example: search_course_content(query="Creating An MCP Client")
   - CRITICAL: Use the EXACT title from Round 1, do not modify or rephrase it

Information Flow:
- Lesson Query: User Input → Content Search → Course Context → Enhanced Answer
- Topic Query: User Input → Lesson Discovery → Topic Search → Comparison Answer

== FAILURE HANDLING PROTOCOL ==
If Round 1 content search fails (no course found):
- Try a broader search with different terms
- Try to search for courses that might contain the query terms in descriptions
- If all attempts fail, accept that no matching course exists and return accordingly

== SYSTEM INTELLIGENCE REQUIREMENTS ==
- Extract the full course title from Round 1 results and use that exact title for Round 2
- Always prioritize finding actual content before structure (Content-First approach)"""
    
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
    
    def _execute_tool_round(self, messages: List[Dict[str, Any]], tools: Optional[List], 
                           tool_manager, round_num: int) -> ToolRoundResult:
        """Execute one complete round of tool calling"""
        logger.info(f"Executing tool round {round_num}")
        
        # API call with current message state
        try:
            api_params = {
                "model": self.model,
                "temperature": 0,
                "max_tokens": 800,
                "messages": messages
            }
            
            # Add tools if available
            if tools:
                openai_tools = self._convert_anthropic_tools_to_openai(tools)
                api_params["tools"] = openai_tools
                api_params["tool_choice"] = "auto"
            
            response = self.client.chat.completions.create(**api_params)
            
        except Exception as e:
            logger.error(f"API error in round {round_num}: {str(e)}")
            return ToolRoundResult(
                round_number=round_num,
                had_tool_calls=False,
                tool_results=[],
                queries_executed=[],
                should_stop=True,
                error=f"API error in round {round_num}: {str(e)}"
            )
        
        # Process response
        message = response.choices[0].message
        
        # No tool calls - conversation is complete
        if not message.tool_calls:
            logger.info(f"Round {round_num} - AI decided no tools needed")
            logger.info(f"Round {round_num} - AI response: {message.content[:200]}...")
            # Add assistant message and stop
            messages.append({"role": "assistant", "content": message.content})
            return ToolRoundResult(
                round_number=round_num,
                had_tool_calls=False,
                tool_results=[],
                queries_executed=[],
                should_stop=True
            )
        
        # Process tool calls
        return self._process_tool_calls_for_round(response, messages, tool_manager, round_num)
    
    def _process_tool_calls_for_round(self, response, messages: List[Dict[str, Any]], 
                                     tool_manager, round_num: int) -> ToolRoundResult:
        """Handle tool execution within a single round"""
        tool_results = []
        queries_executed = []
        
        # Add AI's tool call response to messages
        assistant_message = {
            "role": "assistant", 
            "content": response.choices[0].message.content,
            "tool_calls": response.choices[0].message.tool_calls
        }
        messages.append(assistant_message)
        
        # Execute all tool calls and add results
        for tool_call in response.choices[0].message.tool_calls:
            try:
                import json
                logger.info(f"Round {round_num} - Tool call: {tool_call.function.name}")
                logger.info(f"Raw arguments: {tool_call.function.arguments}")
                
                function_args = json.loads(tool_call.function.arguments)
                logger.info(f"Parsed arguments: {function_args}")
                
                # Extract query for tracking
                if "query" in function_args:
                    queries_executed.append(function_args["query"])
                
                # Execute the tool
                tool_result = tool_manager.execute_tool(
                    tool_call.function.name, 
                    **function_args
                )
                logger.info(f"Tool result: {tool_result[:200]}...")
                
                # Collect successful results
                if tool_result and not tool_result.startswith("Tool execution failed"):
                    tool_results.append(tool_result)
                
                # Add tool result message
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
                
            except Exception as e:
                logger.error(f"Error executing tool {tool_call.function.name} in round {round_num}: {str(e)}")
                error_message = f"Tool execution failed: {str(e)}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": error_message
                })
        
        return ToolRoundResult(
            round_number=round_num,
            had_tool_calls=True,
            tool_results=tool_results,
            queries_executed=queries_executed,
            should_stop=False
        )
    
    def _should_continue_rounds(self, round_results: List[ToolRoundResult], 
                               current_round: int) -> bool:
        """Determine if another round of tool calling is needed"""
        # Max rounds reached
        if current_round >= config.MAX_TOOL_ROUNDS:
            logger.info(f"Max rounds ({config.MAX_TOOL_ROUNDS}) reached")
            return False
        
        # No tool calls in last round means AI is done
        if round_results and not round_results[-1].had_tool_calls:
            logger.info("No tool calls in last round, stopping")
            return False
        
        # Check for duplicate queries to prevent loops
        all_queries = [q for r in round_results for q in r.queries_executed]
        if len(all_queries) != len(set(all_queries)):
            logger.warning("Duplicate queries detected, stopping rounds")
            return False
        
        # Check if we have successful results
        recent_successful = [r for r in round_results[-2:] if r.tool_results]
        if len(recent_successful) == 0:
            logger.info("No successful results in recent rounds, stopping")
            return False
            
        # All checks passed - continue to next round
        logger.info(f"Continuing to round {current_round + 1} - previous rounds had successful tool results")
        return True
    
    def _synthesize_final_response(self, messages: List[Dict[str, Any]], 
                                  round_results: List[ToolRoundResult], 
                                  tools: Optional[List] = None) -> str:
        """Generate final response from all rounds of tool results"""
        # Check if we should skip synthesis entirely for this provider
        if self.provider == "anthropic" and config.SKIP_SYNTHESIS_FOR_ANTHROPIC:
            logger.info("Skipping synthesis for Anthropic provider per configuration")
            all_tool_results = []
            for round_result in round_results:
                all_tool_results.extend(round_result.tool_results)
            fallback_query = "your question"
            return self._format_tool_results_directly(all_tool_results, fallback_query)
        
        # Build enhanced synthesis prompt focused on the user's specific question
        # Extract the original user question from messages
        original_query = "your question"
        for message in messages:
            if message["role"] == "user":
                original_query = message["content"]
                break
        
        # Determine query type for specialized synthesis
        is_comparison_query = any(phrase in original_query.lower() for phrase in [
            "same topic", "similar", "other courses", "cover the same", "also cover"
        ])
        
        if is_comparison_query:
            synthesis_instruction = f"""Based on the tool results above, provide a focused answer to this comparison question: "{original_query}"

COMPARISON QUERY SYNTHESIS GUIDELINES:
- FIRST: Clearly identify what topic/lesson is being compared (from Round 1 results)
- SECOND: List other lessons/courses that cover the same topic (from Round 2 results)  
- Organize as: "Yes/No, here are lessons covering the same topic:" followed by a clear list
- For each similar lesson, briefly explain how it relates to the original topic
- Remove any irrelevant content that doesn't match the topic
- Focus on SIMILARITY and RELEVANCE to the original lesson topic
- Keep the response concise and directly comparative"""
        else:
            synthesis_instruction = f"""Based on the tool results above, provide a focused, well-organized answer to this specific question: "{original_query}"

STANDARD SYNTHESIS GUIDELINES:
- Focus ONLY on information that directly answers the user's question
- If the user asked about a specific lesson, don't include full course outlines unless relevant
- Organize information clearly with bullet points or sections where appropriate
- Remove redundant or duplicate information
- Provide a concise, helpful response that directly addresses what was asked
- If the question was about lesson content, focus on the key topics and concepts covered"""
        
        messages.append({"role": "system", "content": synthesis_instruction})
        
        # Make final API call to synthesize response
        final_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800,
            "messages": messages
        }
        
        # Don't include tools in synthesis call to avoid confusion
        logger.info(f"Final synthesis API call - model: {final_params['model']}")
        
        try:
            logger.info("Making final API call to synthesize tool results")
            final_response = self.client.chat.completions.create(**final_params)
            final_content = final_response.choices[0].message.content
            logger.info(f"Final response content type: {type(final_content)}, length: {len(final_content) if final_content else 0}")
            
            # Check if we got a valid response
            if final_content and final_content.strip():
                logger.info("Synthesis successful, returning synthesized response")
                return final_content
            elif config.ENABLE_SYNTHESIS_FALLBACK:
                # Fallback: Generate response directly from tool results
                logger.warning("Final API response was None or empty, using fallback response generation")
                all_tool_results = []
                for round_result in round_results:
                    all_tool_results.extend(round_result.tool_results)
                fallback_response = self._format_tool_results_directly(all_tool_results, original_query)
                logger.info(f"Generated fallback response: {fallback_response[:200]}...")
                return fallback_response
            else:
                return "I apologize, but I couldn't generate a proper response to your question."
                
        except openai.APITimeoutError:
            if config.ENABLE_SYNTHESIS_FALLBACK:
                logger.warning("API timeout, using fallback response generation")
                all_tool_results = []
                for round_result in round_results:
                    all_tool_results.extend(round_result.tool_results)
                return self._format_tool_results_directly(all_tool_results, "your question")
            else:
                return "I apologize, but generating the response took too long. Please try again with a simpler question."
        except Exception as e:
            if config.ENABLE_SYNTHESIS_FALLBACK:
                logger.error(f"Error in final API call: {str(e)}, using fallback response generation")
                all_tool_results = []
                for round_result in round_results:
                    all_tool_results.extend(round_result.tool_results)
                return self._format_tool_results_directly(all_tool_results, "your question")
            else:
                return f"An error occurred while processing the tool results: {str(e)}"
    
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
        """Generate response using OpenAI models with sequential tool calling support"""
        
        # Check if sequential tool calling is enabled
        if not config.ENABLE_SEQUENTIAL_TOOLS or not tools or not tool_manager:
            # Fall back to legacy single-round behavior
            return self._generate_single_round_response(query, conversation_history, tools, tool_manager)
        
        # Build initial messages array
        messages = self._build_initial_messages(query, conversation_history, tools)
        
        # Sequential tool calling loop
        round_results = []
        for round_num in range(1, config.MAX_TOOL_ROUNDS + 1):
            logger.info(f"Starting tool round {round_num}")
            
            round_result = self._execute_tool_round(messages, tools, tool_manager, round_num)
            
            # Check for errors in this round
            if round_result.error:
                logger.error(f"Error in round {round_num}: {round_result.error}")
                if round_results:  # We have some successful results from previous rounds
                    return self._synthesize_final_response(messages, round_results, tools)
                else:  # First round failed, use fallback
                    return self._handle_complete_failure(Exception(round_result.error))
            
            # Add to results
            round_results.append(round_result)
            
            # Check if we should continue
            if round_result.should_stop:
                logger.info(f"Round {round_num} set should_stop=True, ending sequential calling")
                break
            elif not self._should_continue_rounds(round_results, round_num):
                logger.info(f"Round continuation logic decided to stop after round {round_num}")
                break
        
        # Generate final response from all rounds
        if not round_results:
            return "I apologize, but I couldn't process your request."
        
        # If the last round didn't have tool calls, we already have the final response in messages
        last_round = round_results[-1]
        if not last_round.had_tool_calls:
            # Find the last assistant message
            for message in reversed(messages):
                if message["role"] == "assistant" and message.get("content"):
                    return message["content"]
            return "I apologize, but I couldn't generate a proper response."
        
        # Otherwise, synthesize from tool results
        return self._synthesize_final_response(messages, round_results, tools)
    
    def _build_initial_messages(self, query: str, conversation_history: Optional[str], 
                               tools: Optional[List]) -> List[Dict[str, Any]]:
        """Build initial messages array for the conversation"""
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
        
        return messages
    
    def _generate_single_round_response(self, query: str, conversation_history: Optional[str] = None,
                                      tools: Optional[List] = None, tool_manager=None) -> str:
        """Legacy single-round response generation for backward compatibility"""
        # Build messages array (OpenAI format)
        messages = self._build_initial_messages(query, conversation_history, tools)
        
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
    
    def _handle_complete_failure(self, error: Exception) -> str:
        """Handle complete failure when no rounds succeeded"""
        logger.error(f"Complete failure in tool calling: {str(error)}")
        if isinstance(error, openai.APITimeoutError):
            return "I apologize, but the response took longer than expected to generate. Please try asking a simpler question or try again later."
        elif isinstance(error, openai.APIConnectionError):
            return "I'm having trouble connecting to the language model. Please ensure the LiteLLM proxy server is running at the configured URL."
        else:
            return f"An error occurred while generating the response: {str(error)}"
    
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
        This is a fallback method with improved content filtering and organization.
        """
        if not tool_results or all(not result or result.strip() == "" for result in tool_results):
            return "I couldn't find specific information about your question in the course materials."
        
        # Filter and organize results based on query type
        filtered_results = self._filter_relevant_content(tool_results, query)
        
        if not filtered_results:
            return "I couldn't find specific information about your question in the course materials."
        
        # Format the response based on query type
        if "lesson" in query.lower() and any(char.isdigit() for char in query):
            return self._format_lesson_specific_response(filtered_results, query)
        else:
            return self._format_general_response(filtered_results, query)
    
    def _filter_relevant_content(self, tool_results: List[str], query: str) -> List[str]:
        """Filter tool results to keep only relevant content based on the query"""
        filtered = []
        
        # Check if this is a lesson-specific query
        is_lesson_query = "lesson" in query.lower() and any(char.isdigit() for char in query)
        
        for result in tool_results:
            if not result or not result.strip():
                continue
                
            result = result.strip()
            
            # For lesson-specific queries, prioritize lesson content over course outlines
            if is_lesson_query:
                # Skip full course outlines (they usually start with course info and have many lessons)
                if "Lesson 0:" in result and "Lesson 1:" in result and "Lesson 2:" in result:
                    # This looks like a full course outline, skip it for lesson-specific queries
                    continue
                # Keep lesson content
                if result.startswith("[") and "Lesson" in result:
                    filtered.append(result)
                elif "Course Title:" in result and len(result.split("Lesson")) <= 3:
                    # Keep brief course info but not full outlines
                    lines = result.split('\n')
                    brief_info = []
                    for line in lines[:3]:  # Just first few lines
                        if line.startswith("Course Title:") or line.startswith("Course Instructor:"):
                            brief_info.append(line)
                    if brief_info:
                        filtered.append('\n'.join(brief_info))
            else:
                # For general queries, keep all relevant content
                filtered.append(result)
        
        return filtered
    
    def _format_lesson_specific_response(self, filtered_results: List[str], query: str) -> str:
        """Format response specifically for lesson queries"""
        lesson_content = []
        course_info = ""
        
        for result in filtered_results:
            if result.startswith("[") and "Lesson" in result:
                # This is lesson content
                lesson_content.append(result)
            elif "Course Title:" in result:
                # This is course info
                course_info = result
        
        response_parts = []
        
        # Add brief course context if available
        if course_info:
            response_parts.append(course_info)
        
        # Add lesson content
        if lesson_content:
            response_parts.extend(lesson_content)
        
        if not response_parts:
            return "I couldn't find specific information about the requested lesson."
        
        return "\n\n".join(response_parts)
    
    def _format_general_response(self, filtered_results: List[str], query: str) -> str:
        """Format response for general queries"""
        response = f"Based on the course materials, here's what I found about {query}:\n\n"
        response += "\n\n".join(filtered_results)
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