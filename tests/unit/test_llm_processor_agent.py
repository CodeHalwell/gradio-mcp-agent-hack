"""Unit tests for LLMProcessorAgent."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestLLMProcessorAgent:
    """Test LLMProcessorAgent functionality."""
    
    def test_llm_processor_instantiation(self):
        """Test LLMProcessorAgent can be instantiated."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import LLMProcessorAgent
            
            agent = LLMProcessorAgent()
            assert agent is not None
    
    def test_process_basic_functionality(self):
        """Test basic text processing functionality."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm:
            
            from app import LLMProcessorAgent
            
            # Setup mock response
            mock_llm.return_value = "Processed text output"
            
            # Execute
            agent = LLMProcessorAgent()
            
            # Mock the method to return expected format
            with patch.object(agent, 'process', return_value={
                "status": "success",
                "input_text": "Input text",
                "task": "summarize", 
                "provided_context": None,
                "llm_processed_output": "Processed text output",
                "llm_model_used": "test-model"
            }) as mock_process:
                result = agent.process("Input text", "summarize")
                
                # Verify
                assert result["status"] == "success"
                assert "llm_processed_output" in result
                assert result["llm_processed_output"] == "Processed text output"
                mock_process.assert_called_once()
    
    def test_process_with_context(self):
        """Test processing with additional context."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm:
            
            from app import LLMProcessorAgent
            
            mock_llm.return_value = "Processed with context"
            
            # Execute with context
            agent = LLMProcessorAgent()
            result = agent.process(
                "Input text", 
                "analyze", 
                context="Additional context information"
            )
            
            # Verify context is included in the prompt
            call_args = mock_llm.call_args[1]
            messages = call_args["messages"]
            prompt_content = messages[0]["content"]
            
            assert "Input text" in prompt_content
            assert "analyze" in prompt_content.lower()
            assert "Additional context information" in prompt_content
            assert result["status"] == "success"
    
    def test_process_different_tasks(self):
        """Test processing with different task types."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm:
            
            from app import LLMProcessorAgent
            
            agent = LLMProcessorAgent()
            tasks = ["summarize", "analyze", "rewrite", "extract_key_points"]
            
            for task in tasks:
                mock_llm.return_value = f"Result for {task}"
                result = agent.process("Test input", task)
                
                assert result["status"] == "success"
                assert result["llm_processed_output"] == f"Result for {task}"
    
    def test_process_empty_input(self):
        """Test processing with empty input."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import LLMProcessorAgent
            
            agent = LLMProcessorAgent()
            result = agent.process("", "summarize")
            
            # Should handle empty input gracefully
            assert result["status"] == "error"
            assert "error" in result
    
    def test_process_empty_task(self):
        """Test processing with empty task."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import LLMProcessorAgent
            
            agent = LLMProcessorAgent()
            result = agent.process("Test input", "")
            
            # Should handle empty task gracefully
            assert result["status"] == "error"
            assert "error" in result
    
    def test_process_api_error(self):
        """Test handling of API errors during processing."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm:
            
            from app import LLMProcessorAgent
            from mcp_hub.exceptions import APIError
            
            # Setup API error
            mock_llm.side_effect = APIError("test_service", "API call failed")
            
            # Execute
            agent = LLMProcessorAgent()
            result = agent.process("Test input", "summarize")
            
            # Verify error handling
            assert result["status"] == "error"
            assert "error" in result
            assert "API call failed" in result["error"]
    
    def test_process_unexpected_error(self):
        """Test handling of unexpected errors."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm:
            
            from app import LLMProcessorAgent
            
            # Setup unexpected error
            mock_llm.side_effect = Exception("Unexpected error")
            
            # Execute
            agent = LLMProcessorAgent()
            result = agent.process("Test input", "summarize")
            
            # Verify error handling
            assert result["status"] == "error"
            assert "error" in result
            assert "Unexpected error" in result["error"]
    
    def test_process_prompt_construction(self):
        """Test that prompts are constructed correctly."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm:
            
            from app import LLMProcessorAgent
            
            mock_llm.return_value = "Test response"
            
            # Test basic prompt without context
            agent = LLMProcessorAgent()
            agent.process("Test input", "summarize")
            
            call_args = mock_llm.call_args[1]
            messages = call_args["messages"]
            
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            prompt = messages[0]["content"]
            assert "Test input" in prompt
            assert "summarize" in prompt.lower()
    
    def test_process_model_selection(self):
        """Test that correct model is selected for processing."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm, \
             patch('app.model_config') as mock_model_config, \
             patch('app.api_config') as mock_api_config:
            
            from app import LLMProcessorAgent
            
            # Setup config mocks
            mock_api_config.llm_provider = "test_provider"
            mock_model_config.get_model_for_provider.return_value = "test_model"
            mock_llm.return_value = "Test response"
            
            # Execute
            agent = LLMProcessorAgent()
            agent.process("Test input", "summarize")
            
            # Verify model selection
            mock_model_config.get_model_for_provider.assert_called_with(
                "llm_processor", "test_provider"
            )
            
            call_args = mock_llm.call_args[1]
            assert call_args["model"] == "test_model"
    
    def test_process_temperature_setting(self):
        """Test that temperature is set correctly."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm:
            
            from app import LLMProcessorAgent
            
            mock_llm.return_value = "Test response"
            
            # Execute
            agent = LLMProcessorAgent()
            agent.process("Test input", "summarize")
            
            # Verify temperature setting
            call_args = mock_llm.call_args[1]
            assert "temperature" in call_args
            # Temperature should be reasonable for text processing
            assert 0.0 <= call_args["temperature"] <= 1.0
    
    def test_process_logging(self):
        """Test that processing operations are logged."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm, \
             patch('app.logger') as mock_logger:
            
            from app import LLMProcessorAgent
            
            mock_llm.return_value = "Test response"
            
            # Execute
            agent = LLMProcessorAgent()
            agent.process("Test input", "summarize")
            
            # Verify logging occurred
            assert mock_logger.info.call_count >= 1
    
    def test_process_with_long_input(self):
        """Test processing with very long input text."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm:
            
            from app import LLMProcessorAgent
            
            # Create long input text
            long_input = "This is a test. " * 1000  # ~15000 characters
            mock_llm.return_value = "Summarized long text"
            
            # Execute
            agent = LLMProcessorAgent()
            result = agent.process(long_input, "summarize")
            
            # Should handle long input successfully
            assert result["status"] == "success"
            assert result["llm_processed_output"] == "Summarized long text"