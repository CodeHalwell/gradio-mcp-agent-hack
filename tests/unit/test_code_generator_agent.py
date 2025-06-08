"""Unit tests for CodeGeneratorAgent."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestCodeGeneratorAgent:
    """Test CodeGeneratorAgent functionality."""
    
    def test_code_generator_instantiation(self):
        """Test CodeGeneratorAgent can be instantiated."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import CodeGeneratorAgent
            
            agent = CodeGeneratorAgent()
            assert agent is not None
    
    def test_generate_code_basic(self):
        """Test basic code generation functionality."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm, \
             patch('app.extract_json_from_text') as mock_extract:
            
            from app import CodeGeneratorAgent
            
            # Setup mock response
            expected_response = {
                "code": "import pandas as pd\\ndf = pd.read_csv('data.csv')\\nprint(df.head())",
                "explanation": "This code reads a CSV file and displays the first 5 rows",
                "dependencies": ["pandas"]
            }
            mock_extract.return_value = expected_response
            mock_llm.return_value = '{"code": "import pandas as pd...", "explanation": "...", "dependencies": ["pandas"]}'
            
            # Execute
            agent = CodeGeneratorAgent()
            result = agent.generate_code("Read a CSV file", "Context about CSV analysis")
            
            # Verify
            assert result[0]["status"] == "success"
            assert "code" in result[0]
            assert "explanation" in result[0]
            assert result[0]["code"] == expected_response["code"]
            mock_llm.assert_called_once()
    
    def test_generate_code_with_context(self):
        """Test code generation with grounded context."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm, \
             patch('app.extract_json_from_text') as mock_extract:
            
            from app import CodeGeneratorAgent
            
            mock_extract.return_value = {
                "code": "# Context-aware code",
                "explanation": "Code based on context",
                "dependencies": []
            }
            mock_llm.return_value = "{}"
            
            # Execute with detailed context
            agent = CodeGeneratorAgent()
            agent.generate_code(
                "Create a data visualization",
                "Use matplotlib to create bar charts. Data comes from pandas DataFrame."
            )
            
            # Verify context is included in prompt
            call_args = mock_llm.call_args[1]
            messages = call_args["messages"]
            prompt_content = messages[0]["content"]
            
            assert "Create a data visualization" in prompt_content
            assert "matplotlib" in prompt_content
            assert "pandas DataFrame" in prompt_content
    
    def test_generate_code_empty_request(self):
        """Test code generation with empty request."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import CodeGeneratorAgent
            
            # Execute with empty request
            agent = CodeGeneratorAgent()
            result = agent.generate_code("", "Some context")
            
            # Verify error handling
            assert result[0]["status"] == "error"
            assert "error" in result[0]
    
    def test_generate_code_invalid_json_response(self):
        """Test handling of invalid JSON response from LLM."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm, \
             patch('app.extract_json_from_text') as mock_extract:
            
            from app import CodeGeneratorAgent
            import json
            
            # Setup invalid JSON scenario
            mock_llm.return_value = "Invalid JSON response"
            mock_extract.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            
            # Execute
            agent = CodeGeneratorAgent()
            result = agent.generate_code("Test request", "Context")
            
            # Verify error handling
            assert result[0]["status"] == "error"
            assert "error" in result[0]
    
    def test_generate_code_missing_required_fields(self):
        """Test handling of response missing required fields."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm, \
             patch('app.extract_json_from_text') as mock_extract:
            
            from app import CodeGeneratorAgent
            from mcp_hub.exceptions import ValidationError
            
            # Setup missing fields scenario
            mock_extract.return_value = {"explanation": "Missing code field"}
            mock_llm.return_value = "{}"
            
            # Execute
            agent = CodeGeneratorAgent()
            result = agent.generate_code("Test request", "Context")
            
            # Verify error handling
            assert result[0]["status"] == "error"
            assert "error" in result[0]
    
    def test_generate_code_api_error(self):
        """Test handling of API errors during code generation."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm:
            
            from app import CodeGeneratorAgent
            from mcp_hub.exceptions import APIError
            
            # Setup API error
            mock_llm.side_effect = APIError("test_service", "API call failed")
            
            # Execute
            agent = CodeGeneratorAgent()
            result = agent.generate_code("Test request", "Context")
            
            # Verify error handling
            assert result[0]["status"] == "error"
            assert "error" in result[0]
            assert "API call failed" in result[0]["error"]
    
    def test_generate_code_retry_mechanism(self):
        """Test retry mechanism for code generation."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm, \
             patch('app.extract_json_from_text') as mock_extract, \
             patch('app.app_config') as mock_app_config:
            
            from app import CodeGeneratorAgent
            
            # Setup retry configuration
            mock_app_config.max_code_generation_attempts = 3
            
            # Setup first two attempts to fail, third to succeed
            mock_extract.side_effect = [
                Exception("First attempt fails"),
                Exception("Second attempt fails"),
                {
                    "code": "successful_code",
                    "explanation": "Third attempt succeeds",
                    "dependencies": []
                }
            ]
            mock_llm.return_value = "{}"
            
            # Execute
            agent = CodeGeneratorAgent()
            result = agent.generate_code("Test request", "Context")
            
            # Verify retry worked
            assert mock_llm.call_count == 3
            assert result[0]["status"] == "success"
            assert result[0]["code"] == "successful_code"
    
    def test_generate_code_max_retries_exceeded(self):
        """Test behavior when max retries are exceeded."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm, \
             patch('app.extract_json_from_text') as mock_extract, \
             patch('app.app_config') as mock_app_config:
            
            from app import CodeGeneratorAgent
            
            # Setup retry configuration
            mock_app_config.max_code_generation_attempts = 2
            
            # Setup all attempts to fail
            mock_extract.side_effect = Exception("All attempts fail")
            mock_llm.return_value = "{}"
            
            # Execute
            agent = CodeGeneratorAgent()
            result = agent.generate_code("Test request", "Context")
            
            # Verify max retries behavior
            assert mock_llm.call_count == 2
            assert result[0]["status"] == "error"
            assert "error" in result[0]
    
    def test_generate_code_model_selection(self):
        """Test that correct model is selected for code generation."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm, \
             patch('app.extract_json_from_text') as mock_extract, \
             patch('app.model_config') as mock_model_config, \
             patch('app.api_config') as mock_api_config:
            
            from app import CodeGeneratorAgent
            
            # Setup config mocks
            mock_api_config.llm_provider = "test_provider"
            mock_model_config.get_model_for_provider.return_value = "code_model"
            mock_extract.return_value = {
                "code": "test_code",
                "explanation": "test_explanation",
                "dependencies": []
            }
            mock_llm.return_value = "{}"
            
            # Execute
            agent = CodeGeneratorAgent()
            agent.generate_code("Test request", "Context")
            
            # Verify model selection
            mock_model_config.get_model_for_provider.assert_called_with(
                "code_generator", "test_provider"
            )
            
            call_args = mock_llm.call_args[1]
            assert call_args["model"] == "code_model"
    
    def test_generate_code_temperature_setting(self):
        """Test that appropriate temperature is set for code generation."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm, \
             patch('app.extract_json_from_text') as mock_extract, \
             patch('app.app_config') as mock_app_config:
            
            from app import CodeGeneratorAgent
            
            # Setup config
            mock_app_config.code_gen_temperature = 0.1
            mock_extract.return_value = {
                "code": "test_code",
                "explanation": "test_explanation",
                "dependencies": []
            }
            mock_llm.return_value = "{}"
            
            # Execute
            agent = CodeGeneratorAgent()
            agent.generate_code("Test request", "Context")
            
            # Verify temperature setting
            call_args = mock_llm.call_args[1]
            assert call_args["temperature"] == 0.1
    
    def test_generate_code_prompt_structure(self):
        """Test that the prompt is structured correctly for code generation."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm, \
             patch('app.extract_json_from_text') as mock_extract:
            
            from app import CodeGeneratorAgent
            
            mock_extract.return_value = {
                "code": "test_code",
                "explanation": "test_explanation",
                "dependencies": []
            }
            mock_llm.return_value = "{}"
            
            # Execute
            agent = CodeGeneratorAgent()
            agent.generate_code("Create a web scraper", "Use requests and BeautifulSoup")
            
            # Verify prompt structure
            call_args = mock_llm.call_args[1]
            messages = call_args["messages"]
            
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            prompt = messages[0]["content"]
            
            # Check prompt contains key elements
            assert "Create a web scraper" in prompt
            assert "requests" in prompt
            assert "BeautifulSoup" in prompt
            assert "JSON" in prompt.upper()
            assert "code" in prompt.lower()
            assert "explanation" in prompt.lower()
    
    def test_generate_code_response_format(self):
        """Test that correct response format is specified."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm, \
             patch('app.extract_json_from_text') as mock_extract:
            
            from app import CodeGeneratorAgent
            
            mock_extract.return_value = {
                "code": "test_code",
                "explanation": "test_explanation",
                "dependencies": []
            }
            mock_llm.return_value = "{}"
            
            # Execute
            agent = CodeGeneratorAgent()
            agent.generate_code("Test request", "Context")
            
            # Verify response format
            call_args = mock_llm.call_args[1]
            response_format = call_args["response_format"]
            
            assert response_format["type"] == "json_object"
            assert "code" in response_format["object"]
            assert "explanation" in response_format["object"]
            assert "dependencies" in response_format["object"]
    
    def test_generate_code_logging(self):
        """Test that code generation operations are logged."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.make_llm_completion') as mock_llm, \
             patch('app.extract_json_from_text') as mock_extract, \
             patch('app.logger') as mock_logger:
            
            from app import CodeGeneratorAgent
            
            mock_extract.return_value = {
                "code": "test_code",
                "explanation": "test_explanation",
                "dependencies": []
            }
            mock_llm.return_value = "{}"
            
            # Execute
            agent = CodeGeneratorAgent()
            agent.generate_code("Test request", "Context")
            
            # Verify logging occurred
            assert mock_logger.info.call_count >= 1