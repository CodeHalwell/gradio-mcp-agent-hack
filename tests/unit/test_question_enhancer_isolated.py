"""Isolated unit tests for QuestionEnhancerAgent."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestQuestionEnhancerAgentIsolated:
    """Test QuestionEnhancerAgent in isolation with minimal dependencies."""
    
    def test_import_agent_class(self):
        """Test that we can import and instantiate the agent class."""
        # Mock all external dependencies
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm, \
             patch('app.validate_non_empty_string'):
            
            # Import only after mocking
            from app import QuestionEnhancerAgent
            
            # Test instantiation
            agent = QuestionEnhancerAgent()
            assert agent is not None
    
    def test_enhance_question_basic_functionality(self):
        """Test basic enhance_question functionality with mocked dependencies."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm, \
             patch('app.validate_non_empty_string') as mock_validate, \
             patch('app.extract_json_from_text') as mock_extract:
            
            from app import QuestionEnhancerAgent
            
            # Setup mocks
            expected_response = {
                "sub_questions": [
                    "How to read CSV files in Python?",
                    "What are the best libraries for CSV data analysis?", 
                    "How to create visualizations from CSV data?"
                ]
            }
            mock_extract.return_value = expected_response
            mock_llm.return_value = json.dumps(expected_response)
            
            # Execute
            agent = QuestionEnhancerAgent()
            result = agent.enhance_question("Test request", num_questions=3)
            
            # Verify
            assert result == expected_response
            assert len(result["sub_questions"]) == 3
            mock_validate.assert_called_once()
            mock_llm.assert_called_once()
    
    def test_enhance_question_error_handling(self):
        """Test error handling in enhance_question."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm, \
             patch('app.validate_non_empty_string') as mock_validate:
            
            from app import QuestionEnhancerAgent
            from mcp_hub.exceptions import ValidationError
            
            # Setup error scenario
            mock_validate.side_effect = ValidationError("Empty request")
            
            # Execute
            agent = QuestionEnhancerAgent()
            result = agent.enhance_question("", num_questions=3)
            
            # Verify error handling
            assert "error" in result
            assert result["sub_questions"] == []
    
    def test_enhance_question_invalid_json(self):
        """Test handling of invalid JSON from LLM."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm, \
             patch('app.validate_non_empty_string'), \
             patch('app.extract_json_from_text') as mock_extract:
            
            from app import QuestionEnhancerAgent
            
            # Setup invalid JSON scenario
            mock_extract.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            
            # Execute
            agent = QuestionEnhancerAgent()
            result = agent.enhance_question("Test request", num_questions=3)
            
            # Verify error handling
            assert "error" in result
            assert result["sub_questions"] == []
    
    def test_enhance_question_missing_key(self):
        """Test handling of missing sub_questions key."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm, \
             patch('app.validate_non_empty_string'), \
             patch('app.extract_json_from_text') as mock_extract:
            
            from app import QuestionEnhancerAgent
            from mcp_hub.exceptions import ValidationError
            
            # Setup missing key scenario
            mock_extract.return_value = {"wrong_key": ["Question 1"]}
            
            # Execute
            agent = QuestionEnhancerAgent()
            result = agent.enhance_question("Test request", num_questions=3)
            
            # Verify error handling
            assert "error" in result
            assert result["sub_questions"] == []