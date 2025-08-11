#!/usr/bin/env python3
"""
Unit tests for sequential tool calling functionality in AIGenerator
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add backend to path for imports
sys.path.append('backend')

from ai_generator import AIGenerator, ToolRoundResult
from config import config
from search_tools import ToolManager

class TestSequentialToolCalling(unittest.TestCase):
    """Test sequential tool calling behavior"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ai_generator = AIGenerator()
        self.mock_tool_manager = Mock(spec=ToolManager)
        self.mock_tools = [{
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }]
        
        # Enable sequential tools for testing
        config.ENABLE_SEQUENTIAL_TOOLS = True
        config.MAX_TOOL_ROUNDS = 2
        config.ENABLE_SYNTHESIS_FALLBACK = False  # Disable fallback for cleaner testing
    
    @patch('ai_generator.openai.OpenAI')
    def test_single_round_sufficient(self, mock_openai_client):
        """Test when AI gets answer in round 1 and doesn't need round 2"""
        # Mock the AI response without tool calls (indicating completion)
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.content = "Here's the answer to your question."
        
        # Set up the mock client
        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai_client.return_value = mock_client_instance
        self.ai_generator.client = mock_client_instance
        
        # Set the provider to ensure proper behavior
        self.ai_generator.provider = "openai"
        
        # Call the method
        result = self.ai_generator.generate_response(
            "What is MCP?",
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify results
        self.assertEqual(result, "Here's the answer to your question.")
        # Should only make one API call since no tools were used
        self.assertEqual(mock_client_instance.chat.completions.create.call_count, 1)
        # Tool manager should not be called since no tool calls were made
        self.mock_tool_manager.execute_tool.assert_not_called()
    
    @patch('ai_generator.openai.OpenAI')
    def test_two_rounds_needed(self, mock_openai_client):
        """Test when Round 1 partial results, Round 2 completes the answer"""
        # Set up mock responses for two rounds
        mock_client_instance = Mock()
        
        # Round 1: AI makes tool calls
        mock_tool_call_1 = Mock()
        mock_tool_call_1.id = "call_1"
        mock_tool_call_1.function.name = "search_course_content"
        mock_tool_call_1.function.arguments = '{"query": "MCP basics"}'
        
        mock_response_1 = Mock()
        mock_response_1.choices = [Mock()]
        mock_response_1.choices[0].message = Mock()
        mock_response_1.choices[0].message.tool_calls = [mock_tool_call_1]
        mock_response_1.choices[0].message.content = None
        
        # Round 2: AI makes more tool calls
        mock_tool_call_2 = Mock()
        mock_tool_call_2.id = "call_2"
        mock_tool_call_2.function.name = "search_course_content"
        mock_tool_call_2.function.arguments = '{"query": "MCP implementation"}'
        
        mock_response_2 = Mock()
        mock_response_2.choices = [Mock()]
        mock_response_2.choices[0].message = Mock()
        mock_response_2.choices[0].message.tool_calls = [mock_tool_call_2]
        mock_response_2.choices[0].message.content = None
        
        # Final synthesis call: AI provides final answer
        mock_response_final = Mock()
        mock_response_final.choices = [Mock()]
        mock_response_final.choices[0].message = Mock()
        mock_response_final.choices[0].message.tool_calls = None
        mock_response_final.choices[0].message.content = "MCP is a protocol for connecting AI assistants to external tools and data sources. Here's how to implement it..."
        
        # Set up the mock to return different responses for each call
        mock_client_instance.chat.completions.create.side_effect = [
            mock_response_1,
            mock_response_2,
            mock_response_final
        ]
        
        # Set up tool manager responses
        self.mock_tool_manager.execute_tool.side_effect = [
            "Basic MCP information found",
            "Implementation details found"
        ]
        
        mock_openai_client.return_value = mock_client_instance
        self.ai_generator.client = mock_client_instance
        self.ai_generator.provider = "openai"
        
        # Call the method
        result = self.ai_generator.generate_response(
            "What is MCP and how do I implement it?",
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify results
        self.assertEqual(result, "MCP is a protocol for connecting AI assistants to external tools and data sources. Here's how to implement it...")
        # Should make 3 API calls (2 rounds + 1 synthesis)
        self.assertEqual(mock_client_instance.chat.completions.create.call_count, 3)
        # Tool manager should be called twice (once per round)
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 2)
    
    @patch('ai_generator.openai.OpenAI')
    def test_max_rounds_enforcement(self, mock_openai_client):
        """Test that system respects MAX_TOOL_ROUNDS limit"""
        # Set low limit for testing
        config.MAX_TOOL_ROUNDS = 2
        
        # Set up mock responses for multiple rounds
        mock_client_instance = Mock()
        
        # Create tool call responses for multiple rounds
        tool_call_responses = []
        for i in range(5):  # Try to create 5 rounds (more than limit)
            mock_tool_call = Mock()
            mock_tool_call.id = f"call_{i}"
            mock_tool_call.function.name = "search_course_content"
            mock_tool_call.function.arguments = f'{{"query": "query {i}"}}'
            
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.tool_calls = [mock_tool_call]
            mock_response.choices[0].message.content = None
            
            tool_call_responses.append(mock_response)
        
        # Final synthesis response
        mock_response_final = Mock()
        mock_response_final.choices = [Mock()]
        mock_response_final.choices[0].message = Mock()
        mock_response_final.choices[0].message.tool_calls = None
        mock_response_final.choices[0].message.content = "Final answer after 2 rounds"
        
        # Set up responses (should only use first 2 + synthesis)
        mock_client_instance.chat.completions.create.side_effect = [
            tool_call_responses[0],  # Round 1
            tool_call_responses[1],  # Round 2
            mock_response_final      # Synthesis
        ]
        
        # Set up tool manager responses
        self.mock_tool_manager.execute_tool.return_value = "Search result"
        
        mock_openai_client.return_value = mock_client_instance
        self.ai_generator.client = mock_client_instance
        self.ai_generator.provider = "openai"
        
        # Call the method
        result = self.ai_generator.generate_response(
            "Complex multi-part question",
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify results
        self.assertEqual(result, "Final answer after 2 rounds")
        # Should make exactly 3 API calls (2 rounds + 1 synthesis)
        self.assertEqual(mock_client_instance.chat.completions.create.call_count, 3)
        # Tool manager should be called exactly twice (once per allowed round)
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 2)
    
    @patch('ai_generator.openai.OpenAI')
    def test_duplicate_query_detection(self, mock_openai_client):
        """Test that system prevents AI from making duplicate searches"""
        mock_client_instance = Mock()
        
        # Round 1: AI makes initial tool call
        mock_tool_call_1 = Mock()
        mock_tool_call_1.id = "call_1"
        mock_tool_call_1.function.name = "search_course_content"
        mock_tool_call_1.function.arguments = '{"query": "MCP protocol"}'
        
        mock_response_1 = Mock()
        mock_response_1.choices = [Mock()]
        mock_response_1.choices[0].message = Mock()
        mock_response_1.choices[0].message.tool_calls = [mock_tool_call_1]
        mock_response_1.choices[0].message.content = None
        
        # Round 2: AI tries to make same query again (should be detected)
        mock_tool_call_2 = Mock()
        mock_tool_call_2.id = "call_2"
        mock_tool_call_2.function.name = "search_course_content"
        mock_tool_call_2.function.arguments = '{"query": "MCP protocol"}'  # Same query!
        
        mock_response_2 = Mock()
        mock_response_2.choices = [Mock()]
        mock_response_2.choices[0].message = Mock()
        mock_response_2.choices[0].message.tool_calls = [mock_tool_call_2]
        mock_response_2.choices[0].message.content = None
        
        # Final synthesis response (should be called after duplicate detection)
        mock_response_final = Mock()
        mock_response_final.choices = [Mock()]
        mock_response_final.choices[0].message = Mock()
        mock_response_final.choices[0].message.tool_calls = None
        mock_response_final.choices[0].message.content = "Answer based on first search only"
        
        mock_client_instance.chat.completions.create.side_effect = [
            mock_response_1,
            mock_response_2,
            mock_response_final
        ]
        
        # Set up tool manager response
        self.mock_tool_manager.execute_tool.return_value = "MCP protocol information"
        
        mock_openai_client.return_value = mock_client_instance
        self.ai_generator.client = mock_client_instance
        self.ai_generator.provider = "openai"
        
        # Call the method
        result = self.ai_generator.generate_response(
            "Tell me about MCP protocol",
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify results
        self.assertEqual(result, "Answer based on first search only")
        # Should stop after detecting duplicate and synthesize
        self.assertEqual(mock_client_instance.chat.completions.create.call_count, 3)
        # Tool manager should be called twice (both rounds execute, but then we detect duplicate)
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 2)
    
    @patch('ai_generator.openai.OpenAI')
    def test_api_error_round_1(self, mock_openai_client):
        """Test error handling when first round fails"""
        mock_client_instance = Mock()
        
        # First API call fails
        mock_client_instance.chat.completions.create.side_effect = Exception("API Error")
        
        mock_openai_client.return_value = mock_client_instance
        self.ai_generator.client = mock_client_instance
        self.ai_generator.provider = "openai"
        
        # Call the method
        result = self.ai_generator.generate_response(
            "What is MCP?",
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify error handling
        self.assertIn("error occurred", result.lower())
        # Should only attempt one API call
        self.assertEqual(mock_client_instance.chat.completions.create.call_count, 1)
        # Tool manager should not be called due to API error
        self.mock_tool_manager.execute_tool.assert_not_called()
    
    @patch('ai_generator.openai.OpenAI')
    def test_api_error_round_2_with_fallback(self, mock_openai_client):
        """Test error handling when second round fails but first succeeded"""
        mock_client_instance = Mock()
        
        # Round 1: Successful tool call
        mock_tool_call_1 = Mock()
        mock_tool_call_1.id = "call_1"
        mock_tool_call_1.function.name = "search_course_content"
        mock_tool_call_1.function.arguments = '{"query": "MCP basics"}'
        
        mock_response_1 = Mock()
        mock_response_1.choices = [Mock()]
        mock_response_1.choices[0].message = Mock()
        mock_response_1.choices[0].message.tool_calls = [mock_tool_call_1]
        mock_response_1.choices[0].message.content = None
        
        # Round 2: API error
        mock_client_instance.chat.completions.create.side_effect = [
            mock_response_1,  # First round succeeds
            Exception("API Error Round 2"),  # Second round fails
            Mock()  # Synthesis call (if reached)
        ]
        
        # Set up tool manager response for first round
        self.mock_tool_manager.execute_tool.return_value = "MCP information found"
        
        # Mock the synthesis method to avoid complex setup
        with patch.object(self.ai_generator, '_synthesize_final_response') as mock_synthesize:
            mock_synthesize.return_value = "Response based on first round results"
            
            mock_openai_client.return_value = mock_client_instance
            self.ai_generator.client = mock_client_instance
            self.ai_generator.provider = "openai"
            
            # Call the method
            result = self.ai_generator.generate_response(
                "Complex question about MCP",
                tools=self.mock_tools,
                tool_manager=self.mock_tool_manager
            )
            
            # Verify results
            self.assertEqual(result, "Response based on first round results")
            # Should call synthesis with results from first round
            mock_synthesize.assert_called_once()
    
    def test_message_history_preservation(self):
        """Test that message history is properly maintained across rounds"""
        # Create a real ToolRoundResult to test message building
        round_result = ToolRoundResult(
            round_number=1,
            had_tool_calls=True,
            tool_results=["Test result"],
            queries_executed=["test query"],
            should_stop=False
        )
        
        # Test _build_initial_messages
        messages = self.ai_generator._build_initial_messages(
            "Test query", 
            "Previous conversation", 
            self.mock_tools
        )
        
        # Verify message structure
        self.assertEqual(len(messages), 2)  # system + user
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "Test query")
        self.assertIn("Previous conversation", messages[0]["content"])
        self.assertIn("MULTI-STEP REASONING", messages[0]["content"])
    
    def test_tool_round_result_creation(self):
        """Test ToolRoundResult data class functionality"""
        # Test successful round
        success_result = ToolRoundResult(
            round_number=1,
            had_tool_calls=True,
            tool_results=["Result 1", "Result 2"],
            queries_executed=["query 1", "query 2"],
            should_stop=False
        )
        
        self.assertEqual(success_result.round_number, 1)
        self.assertTrue(success_result.had_tool_calls)
        self.assertEqual(len(success_result.tool_results), 2)
        self.assertEqual(len(success_result.queries_executed), 2)
        self.assertFalse(success_result.should_stop)
        self.assertIsNone(success_result.error)
        
        # Test error result
        error_result = ToolRoundResult(
            round_number=2,
            had_tool_calls=False,
            tool_results=[],
            queries_executed=[],
            should_stop=True,
            error="API timeout"
        )
        
        self.assertEqual(error_result.round_number, 2)
        self.assertFalse(error_result.had_tool_calls)
        self.assertTrue(error_result.should_stop)
        self.assertEqual(error_result.error, "API timeout")
    
    def test_legacy_fallback_mode(self):
        """Test that legacy single-round behavior is preserved when sequential tools disabled"""
        # Disable sequential tools
        config.ENABLE_SEQUENTIAL_TOOLS = False
        
        with patch.object(self.ai_generator, '_generate_single_round_response') as mock_single_round:
            mock_single_round.return_value = "Legacy response"
            
            result = self.ai_generator.generate_response(
                "Test query",
                tools=self.mock_tools,
                tool_manager=self.mock_tool_manager
            )
            
            self.assertEqual(result, "Legacy response")
            mock_single_round.assert_called_once()
    
    def test_no_tools_fallback(self):
        """Test behavior when no tools are provided"""
        with patch.object(self.ai_generator, '_generate_single_round_response') as mock_single_round:
            mock_single_round.return_value = "No tools response"
            
            result = self.ai_generator.generate_response("Test query")
            
            self.assertEqual(result, "No tools response")
            mock_single_round.assert_called_once()


class TestSequentialToolConfiguration(unittest.TestCase):
    """Test configuration options for sequential tool calling"""
    
    def test_config_defaults(self):
        """Test that configuration has appropriate defaults"""
        self.assertEqual(config.MAX_TOOL_ROUNDS, 2)
        self.assertTrue(config.ENABLE_SEQUENTIAL_TOOLS)
    
    def test_config_override(self):
        """Test that configuration can be overridden"""
        original_max_rounds = config.MAX_TOOL_ROUNDS
        original_enable = config.ENABLE_SEQUENTIAL_TOOLS
        
        # Override settings
        config.MAX_TOOL_ROUNDS = 3
        config.ENABLE_SEQUENTIAL_TOOLS = False
        
        self.assertEqual(config.MAX_TOOL_ROUNDS, 3)
        self.assertFalse(config.ENABLE_SEQUENTIAL_TOOLS)
        
        # Restore original settings
        config.MAX_TOOL_ROUNDS = original_max_rounds
        config.ENABLE_SEQUENTIAL_TOOLS = original_enable


if __name__ == "__main__":
    unittest.main()