"""Unit tests for QuestionEnhancerAgent."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Mock dependencies before importing app
with patch('modal.App'):
    with patch('modal.Image'):
        with patch('tavily.TavilyClient'):
            from app import QuestionEnhancerAgent
            from mcp_hub.exceptions import ValidationError, APIError


class TestQuestionEnhancerAgent:
    """Test suite for QuestionEnhancerAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = QuestionEnhancerAgent()
    
    @patch('app.make_llm_completion')
    @patch('app.validate_non_empty_string')
    def test_enhance_question_success(self, mock_validate, mock_llm):
        """Test successful question enhancement."""
        # Setup
        user_request = "How do I analyze CSV data with Python?"
        expected_response = {
            "sub_questions": [
                "How to read CSV files in Python?",
                "What are the best libraries for CSV data analysis?", 
                "How to create visualizations from CSV data?"
            ]
        }
        mock_llm.return_value = json.dumps(expected_response)
        
        # Execute
        result = self.agent.enhance_question(user_request, num_questions=3)
        
        # Verify
        mock_validate.assert_called_once_with(user_request, "User request")
        mock_llm.assert_called_once()
        
        assert result == expected_response
        assert len(result["sub_questions"]) == 3
        assert all(isinstance(q, str) for q in result["sub_questions"])
    
    @patch('app.make_llm_completion')
    @patch('app.validate_non_empty_string')
    def test_enhance_question_different_num_questions(self, mock_validate, mock_llm):
        """Test question enhancement with different numbers of questions."""
        user_request = "Test request"
        expected_response = {
            "sub_questions": [
                "Question 1?",
                "Question 2?",
                "Question 3?",
                "Question 4?",
                "Question 5?"
            ]
        }
        mock_llm.return_value = json.dumps(expected_response)
        
        result = self.agent.enhance_question(user_request, num_questions=5)
        
        assert result == expected_response
        assert len(result["sub_questions"]) == 5
    
    def test_enhance_question_empty_request(self):
        """Test question enhancement with empty request."""
        with patch('app.validate_non_empty_string') as mock_validate:
            mock_validate.side_effect = ValidationError("Empty request")
            
            result = self.agent.enhance_question("", num_questions=3)
            
            assert "error" in result
            assert result["sub_questions"] == []
    
    @patch('app.make_llm_completion')
    @patch('app.validate_non_empty_string')
    def test_enhance_question_invalid_json_response(self, mock_validate, mock_llm):
        """Test handling of invalid JSON response from LLM."""
        mock_llm.return_value = "Invalid JSON response"
        
        result = self.agent.enhance_question("Test request", num_questions=3)
        
        assert "error" in result
        assert result["sub_questions"] == []
    
    @patch('app.make_llm_completion')
    @patch('app.validate_non_empty_string')
    def test_enhance_question_missing_sub_questions_key(self, mock_validate, mock_llm):
        """Test handling of response missing sub_questions key."""
        mock_llm.return_value = json.dumps({"wrong_key": ["Question 1"]})
        
        result = self.agent.enhance_question("Test request", num_questions=3)
        
        assert "error" in result
        assert "sub_questions" not in result or result["sub_questions"] == []
    
    @patch('app.make_llm_completion')
    @patch('app.validate_non_empty_string')
    def test_enhance_question_invalid_sub_questions_format(self, mock_validate, mock_llm):
        """Test handling of invalid sub_questions format."""
        # Test with non-list value
        mock_llm.return_value = json.dumps({"sub_questions": "not a list"})
        
        result = self.agent.enhance_question("Test request", num_questions=3)
        
        assert "error" in result
        assert result["sub_questions"] == []
        
        # Test with list containing non-strings
        mock_llm.return_value = json.dumps({"sub_questions": [1, 2, 3]})
        
        result = self.agent.enhance_question("Test request", num_questions=3)
        
        assert "error" in result
        assert result["sub_questions"] == []
    
    @patch('app.make_llm_completion')
    @patch('app.validate_non_empty_string')
    def test_enhance_question_api_error(self, mock_validate, mock_llm):
        """Test handling of API errors."""
        mock_llm.side_effect = APIError("test_service", "API call failed")
        
        result = self.agent.enhance_question("Test request", num_questions=3)
        
        assert "error" in result
        assert result["sub_questions"] == []
    
    @patch('app.make_llm_completion')
    @patch('app.validate_non_empty_string')
    def test_enhance_question_unexpected_error(self, mock_validate, mock_llm):
        """Test handling of unexpected errors."""
        mock_llm.side_effect = Exception("Unexpected error")
        
        result = self.agent.enhance_question("Test request", num_questions=3)
        
        assert "error" in result
        assert "Unexpected error" in result["error"]
        assert result["sub_questions"] == []
    
    @patch('app.make_llm_completion')
    @patch('app.validate_non_empty_string')
    def test_enhance_question_logging(self, mock_validate, mock_llm):
        """Test that appropriate logging occurs."""
        with patch('app.logger') as mock_logger:
            mock_llm.return_value = json.dumps({"sub_questions": ["Q1", "Q2"]})
            
            self.agent.enhance_question("Test request", num_questions=2)
            
            # Verify logging calls
            assert mock_logger.info.call_count >= 2  # At least start and success logs
    
    @patch('app.model_config')  
    @patch('app.api_config')
    @patch('app.make_llm_completion')
    @patch('app.validate_non_empty_string')
    def test_enhance_question_uses_correct_model_config(self, mock_validate, mock_llm, 
                                                       mock_api_config, mock_model_config):
        """Test that correct model configuration is used."""
        mock_api_config.llm_provider = "test_provider"
        mock_model_config.get_model_for_provider.return_value = "test_model"
        mock_llm.return_value = json.dumps({"sub_questions": ["Q1"]})
        
        self.agent.enhance_question("Test request", num_questions=1)
        
        mock_model_config.get_model_for_provider.assert_called_with(
            "question_enhancer", "test_provider"
        )
        
        # Verify LLM was called with correct model
        call_args = mock_llm.call_args[1]
        assert call_args["model"] == "test_model"
    
    @patch('app.make_llm_completion')
    @patch('app.validate_non_empty_string')
    def test_enhance_question_prompt_structure(self, mock_validate, mock_llm):
        """Test that the prompt is structured correctly."""
        mock_llm.return_value = json.dumps({"sub_questions": ["Q1"]})
        
        self.agent.enhance_question("Test request", num_questions=1)
        
        # Get the messages passed to LLM
        call_args = mock_llm.call_args[1]
        messages = call_args["messages"]
        
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "Test request" in messages[0]["content"]
        assert "sub_questions" in messages[0]["content"]
        assert "JSON" in messages[0]["content"]
    
    @patch('app.make_llm_completion')
    @patch('app.validate_non_empty_string')
    def test_enhance_question_response_format(self, mock_validate, mock_llm):
        """Test that correct response format is specified."""
        mock_llm.return_value = json.dumps({"sub_questions": ["Q1"]})
        
        self.agent.enhance_question("Test request", num_questions=1)
        
        call_args = mock_llm.call_args[1]
        response_format = call_args["response_format"]
        
        assert response_format["type"] == "json_object"
        assert "sub_questions" in response_format["object"]
        assert response_format["object"]["sub_questions"]["type"] == "array"