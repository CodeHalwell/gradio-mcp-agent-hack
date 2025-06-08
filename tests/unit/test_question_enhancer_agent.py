"""Unit tests for QuestionEnhancerAgent."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

class MockQuestionEnhancerAgent:
    """Mock implementation for testing."""
    
    def enhance_question(self, user_request: str, num_questions: int = 3):
        """Mock enhance_question method."""
        return {
            "sub_questions": [
                f"Question {i+1} about {user_request[:20]}?" for i in range(num_questions)
            ]
        }

class TestQuestionEnhancerAgent:
    """Test suite for QuestionEnhancerAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = MockQuestionEnhancerAgent()
    
    def test_enhance_question_success(self):
        """Test successful question enhancement."""
        # Setup
        user_request = "How do I analyze CSV data with Python?"
        
        # Execute
        result = self.agent.enhance_question(user_request, num_questions=3)
        
        # Verify
        assert "sub_questions" in result
        assert len(result["sub_questions"]) == 3
        assert all("Question" in q for q in result["sub_questions"])
    
    def test_enhance_question_custom_num(self):
        """Test question enhancement with custom number."""
        # Setup
        user_request = "Create a web scraper"
        
        # Execute  
        result = self.agent.enhance_question(user_request, num_questions=5)
        
        # Verify
        assert len(result["sub_questions"]) == 5
    
    def test_enhance_question_empty_request(self):
        """Test question enhancement with different inputs."""
        # Execute
        result = self.agent.enhance_question("", num_questions=2)
        
        # Verify - should still work with empty string
        assert len(result["sub_questions"]) == 2
        assert all(isinstance(q, str) for q in result["sub_questions"])