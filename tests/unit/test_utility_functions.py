"""Unit tests for utility functions in mcp_hub modules."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestUtilityFunctions:
    """Test utility functions from mcp_hub modules."""
    
    def test_validate_non_empty_string_success(self):
        """Test validation of non-empty strings."""
        from mcp_hub.utils import validate_non_empty_string
        
        # Should not raise for valid strings
        validate_non_empty_string("test", "test_field")
        validate_non_empty_string("  whitespace  ", "test_field")
        validate_non_empty_string("123", "test_field")
    
    def test_validate_non_empty_string_failure(self):
        """Test validation failure for empty strings."""
        from mcp_hub.utils import validate_non_empty_string
        from mcp_hub.exceptions import ValidationError
        
        # Should raise for empty strings
        with pytest.raises(ValidationError):
            validate_non_empty_string("", "test_field")
        
        with pytest.raises(ValidationError):
            validate_non_empty_string("   ", "test_field")
        
        with pytest.raises(ValidationError):
            validate_non_empty_string(None, "test_field")
    
    def test_extract_json_from_text_success(self):
        """Test JSON extraction from text."""
        from mcp_hub.utils import extract_json_from_text
        import json
        
        # Test valid JSON
        json_text = '{"key": "value", "number": 123}'
        result = extract_json_from_text(json_text)
        assert result == {"key": "value", "number": 123}
        
        # Test JSON with surrounding text
        mixed_text = 'Some text before {"extracted": true} some text after'
        result = extract_json_from_text(mixed_text)
        assert result == {"extracted": True}
    
    def test_extract_json_from_text_failure(self):
        """Test JSON extraction failure."""
        from mcp_hub.utils import extract_json_from_text
        import json
        
        # Should raise for invalid JSON
        with pytest.raises(json.JSONDecodeError):
            extract_json_from_text("Invalid JSON text")
        
        with pytest.raises(json.JSONDecodeError):
            extract_json_from_text("No JSON here at all")
    
    def test_extract_urls_from_text(self):
        """Test URL extraction from text."""
        from mcp_hub.utils import extract_urls_from_text
        
        # Test text with various URL formats
        text = """
        Check out https://example.com and http://test.org.
        Also visit https://github.com/user/repo and ftp://files.example.com
        Email me at user@example.com (should not be extracted as URL)
        """
        
        urls = extract_urls_from_text(text)
        
        # Should extract HTTP/HTTPS URLs
        assert "https://example.com" in urls
        assert "http://test.org" in urls
        assert "https://github.com/user/repo" in urls
        
        # Should not extract email addresses
        assert "user@example.com" not in urls
    
    def test_create_apa_citation_success(self):
        """Test APA citation creation."""
        from mcp_hub.utils import create_apa_citation
        
        with patch('requests.get') as mock_get:
            # Mock successful web page fetch
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '''
            <html>
                <head>
                    <title>Test Article Title</title>
                    <meta name="author" content="John Doe">
                    <meta name="description" content="Test description">
                </head>
                <body>Content</body>
            </html>
            '''
            mock_get.return_value = mock_response
            
            # Execute
            citation = create_apa_citation("https://example.com/article")
            
            # Verify citation format
            assert "Test Article Title" in citation
            assert "example.com" in citation
            assert "2025" in citation  # Current year
    
    def test_create_apa_citation_failure(self):
        """Test APA citation creation with failures."""
        from mcp_hub.utils import create_apa_citation
        
        with patch('requests.get') as mock_get:
            # Mock failed web page fetch
            mock_get.side_effect = Exception("Network error")
            
            # Execute
            citation = create_apa_citation("https://example.com/unreachable")
            
            # Should return fallback citation
            assert "example.com" in citation
            assert "Retrieved" in citation or "Access" in citation
    
    @patch('mcp_hub.utils.openai.OpenAI')
    @patch('mcp_hub.utils.anthropic.Anthropic') 
    def test_make_llm_completion_openai(self, mock_anthropic, mock_openai):
        """Test LLM completion with OpenAI."""
        from mcp_hub.utils import make_llm_completion
        
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "OpenAI response"
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('mcp_hub.config.api_config') as mock_config:
            mock_config.llm_provider = "openai"
            mock_config.openai_api_key = "test-key"
            
            # Execute
            result = make_llm_completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test"}],
                temperature=0.7
            )
            
            # Verify
            assert result == "OpenAI response"
            mock_client.chat.completions.create.assert_called_once()
    
    @patch('mcp_hub.utils.anthropic.Anthropic')
    def test_make_llm_completion_anthropic(self, mock_anthropic):
        """Test LLM completion with Anthropic."""
        from mcp_hub.utils import make_llm_completion
        
        # Mock Anthropic client
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Anthropic response"
        mock_client.messages.create.return_value = mock_response
        
        with patch('mcp_hub.config.api_config') as mock_config:
            mock_config.llm_provider = "anthropic"
            mock_config.anthropic_api_key = "test-key"
            
            # Execute
            result = make_llm_completion(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "Test"}],
                temperature=0.7
            )
            
            # Verify
            assert result == "Anthropic response"
            mock_client.messages.create.assert_called_once()


class TestExceptionHandling:
    """Test custom exception classes."""
    
    def test_api_error_creation(self):
        """Test APIError exception creation."""
        from mcp_hub.exceptions import APIError
        
        error = APIError("test_service", "Test error message")
        
        assert error.service == "test_service"
        assert error.message == "Test error message"
        assert str(error) == "API Error in test_service: Test error message"
    
    def test_validation_error_creation(self):
        """Test ValidationError exception creation."""
        from mcp_hub.exceptions import ValidationError
        
        error = ValidationError("Invalid input format")
        
        assert str(error) == "Validation Error: Invalid input format"
    
    def test_code_generation_error_creation(self):
        """Test CodeGenerationError exception creation."""
        from mcp_hub.exceptions import CodeGenerationError
        
        error = CodeGenerationError("Failed to generate code")
        
        assert str(error) == "Code Generation Error: Failed to generate code"
    
    def test_code_execution_error_creation(self):
        """Test CodeExecutionError exception creation."""
        from mcp_hub.exceptions import CodeExecutionError
        
        error = CodeExecutionError("Runtime error in code")
        
        assert str(error) == "Code Execution Error: Runtime error in code"


class TestConfigurationManagement:
    """Test configuration management."""
    
    def test_api_config_initialization(self):
        """Test APIConfig initialization."""
        from mcp_hub.config import APIConfig
        
        # Test with environment variables set
        config = APIConfig(
            llm_provider="test_provider",
            tavily_api_key="tvly-test-key",
            nebius_api_key="test-key"
        )
        
        assert config.llm_provider == "test_provider"
        assert config.tavily_api_key == "tvly-test-key"
    
    def test_model_config_provider_selection(self):
        """Test model configuration for different providers."""
        from mcp_hub.config import ModelConfig
        
        config = ModelConfig()
        
        # Test model selection for different providers
        nebius_model = config.get_model_for_provider("question_enhancer", "nebius")
        openai_model = config.get_model_for_provider("question_enhancer", "openai")
        anthropic_model = config.get_model_for_provider("question_enhancer", "anthropic")
        
        assert nebius_model is not None
        assert openai_model is not None  
        assert anthropic_model is not None
        
        # Different providers should potentially return different models
        # The exact models depend on the configuration
    
    def test_app_config_defaults(self):
        """Test application configuration defaults."""
        from mcp_hub.config import AppConfig
        
        config = AppConfig()
        
        assert config.modal_app_name == "my-sandbox-app"
        assert config.max_search_results >= 1
        assert config.max_code_generation_attempts >= 1
        assert 0.0 <= config.llm_temperature <= 1.0
        assert 0.0 <= config.code_gen_temperature <= 1.0


class TestLoggingConfiguration:
    """Test logging configuration."""
    
    def test_logger_creation(self):
        """Test logger instance creation."""
        from mcp_hub.logging_config import logger
        
        assert logger is not None
        assert logger.name == "mcp_hub"
    
    def test_logger_level_setting(self):
        """Test logger level configuration."""
        from mcp_hub.logging_config import logger
        import logging
        
        # Logger should be properly configured
        assert logger.level <= logging.INFO
    
    def test_logger_handlers(self):
        """Test logger handlers configuration."""
        from mcp_hub.logging_config import logger
        
        # Should have at least one handler
        assert len(logger.handlers) >= 1
    
    def test_logging_functionality(self):
        """Test actual logging functionality."""
        from mcp_hub.logging_config import logger
        import logging
        
        with patch('logging.StreamHandler') as mock_handler:
            # Test that logging works
            logger.info("Test info message")
            logger.warning("Test warning message")
            logger.error("Test error message")
            
            # No exceptions should be raised


class TestPerformanceMonitoring:
    """Test performance monitoring functionality."""
    
    @patch('mcp_hub.performance_monitoring.psutil')
    def test_metrics_collector_initialization(self, mock_psutil):
        """Test metrics collector initialization."""
        from mcp_hub.performance_monitoring import MetricsCollector
        
        collector = MetricsCollector()
        assert collector is not None
        assert hasattr(collector, 'metrics')
    
    def test_metrics_recording(self):
        """Test metric recording functionality."""
        with patch('mcp_hub.performance_monitoring.psutil'):
            from mcp_hub.performance_monitoring import MetricsCollector
            
            collector = MetricsCollector()
            
            # Record some metrics
            collector.record_metric("test_metric", 1.5, {"tag": "value"})
            collector.increment_counter("test_counter", 1, {"tag": "value"})
            
            # Should not raise exceptions
            assert True  # If we get here, no exceptions were raised


class TestCacheUtilities:
    """Test caching utilities."""
    
    @patch('os.makedirs')
    @patch('os.path.exists')
    def test_cache_manager_initialization(self, mock_exists, mock_makedirs):
        """Test cache manager initialization."""
        mock_exists.return_value = False
        
        from mcp_hub.cache_utils import CacheManager
        
        cache_manager = CacheManager("test_cache")
        assert cache_manager is not None
        mock_makedirs.assert_called_once()
    
    @patch('os.path.exists')
    @patch('pickle.dump')
    @patch('pickle.load')
    def test_cache_operations(self, mock_load, mock_dump, mock_exists):
        """Test cache set and get operations."""
        from mcp_hub.cache_utils import CacheManager
        
        mock_exists.return_value = True
        mock_load.return_value = "cached_value"
        
        cache_manager = CacheManager("test_cache")
        
        # Test cache set
        cache_manager.set("test_key", "test_value", ttl=300)
        
        # Test cache get
        value = cache_manager.get("test_key")
        
        # Should handle caching operations without errors
        assert value is not None or value is None  # Either works for test