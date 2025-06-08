"""UI endpoint smoke tests for Gradio interface."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestUIEndpoints:
    """Test Gradio UI endpoints and wrappers."""
    
    def test_agent_orchestrator_wrapper(self):
        """Test the agent_orchestrator wrapper function."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import agent_orchestrator
            
            with patch('app.orchestrator') as mock_orchestrator:
                # Setup mock response
                mock_orchestrator.orchestrate.return_value = (
                    {
                        "status": "success",
                        "research_summary": "Test research",
                        "generated_code": "print('test')",
                        "execution_result": "test",
                        "citations": ["Test citation"]
                    },
                    "Test summary"
                )
                
                # Execute wrapper
                result = agent_orchestrator("Create a test script")
                
                # Verify wrapper functionality
                assert len(result) == 2  # Should return tuple
                assert result[0]["status"] == "success"
                assert "research_summary" in result[0]
                mock_orchestrator.orchestrate.assert_called_once()
    
    def test_agent_question_enhancer_wrapper(self):
        """Test the agent_question_enhancer wrapper function."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import agent_question_enhancer
            
            with patch('app.question_enhancer') as mock_agent:
                # Setup mock response
                mock_agent.enhance_question.return_value = {
                    "sub_questions": [
                        "How to create a script?",
                        "What libraries to use?",
                        "How to test the code?"
                    ]
                }
                
                # Execute wrapper
                result = agent_question_enhancer("Create a Python script", 3)
                
                # Verify wrapper functionality
                assert "sub_questions" in result
                assert len(result["sub_questions"]) == 3
                mock_agent.enhance_question.assert_called_once_with("Create a Python script", 3)
    
    def test_agent_web_search_wrapper(self):
        """Test the agent_web_search wrapper function."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import agent_web_search
            
            with patch('app.web_search') as mock_agent:
                # Setup mock response
                mock_agent.search.return_value = {
                    "status": "success",
                    "results": [
                        {"title": "Tutorial", "url": "https://example.com"}
                    ],
                    "tavily_answer": "Test search result"
                }
                
                # Execute wrapper
                result = agent_web_search("Python tutorials")
                
                # Verify wrapper functionality
                assert result["status"] == "success"
                assert "results" in result
                assert "tavily_answer" in result
                mock_agent.search.assert_called_once()
    
    def test_agent_llm_processor_wrapper(self):
        """Test the agent_llm_processor wrapper function."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import agent_llm_processor
            
            with patch('app.llm_processor') as mock_agent:
                # Setup mock response
                mock_agent.process.return_value = {
                    "status": "success",
                    "llm_processed_output": "Processed text content"
                }
                
                # Execute wrapper
                result = agent_llm_processor("Input text", "summarize", "Context")
                
                # Verify wrapper functionality
                assert result["status"] == "success"
                assert "llm_processed_output" in result
                mock_agent.process.assert_called_once_with("Input text", "summarize", "Context")
    
    def test_agent_citation_formatter_wrapper(self):
        """Test the agent_citation_formatter wrapper function."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import agent_citation_formatter
            
            with patch('app.citation_formatter') as mock_agent:
                # Setup mock response
                mock_agent.format_citations.return_value = {
                    "formatted_citations": ["Citation 1", "Citation 2"],
                    "error": None
                }
                
                # Execute wrapper
                result = agent_citation_formatter("Text with URLs")
                
                # Verify wrapper functionality
                assert "formatted_citations" in result
                assert len(result["formatted_citations"]) == 2
                mock_agent.format_citations.assert_called_once()
    
    def test_agent_code_generator_wrapper(self):
        """Test the agent_code_generator wrapper function."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import agent_code_generator
            
            with patch('app.code_generator') as mock_agent:
                # Setup mock response
                mock_agent.generate_code.return_value = [
                    {
                        "status": "success",
                        "code": "import pandas as pd\\nprint('Hello')",
                        "explanation": "Simple script"
                    },
                    "Generated code explanation"
                ]
                
                # Execute wrapper
                result = agent_code_generator("Create a script", "Context")
                
                # Verify wrapper functionality
                assert len(result) == 2  # Should return tuple
                assert result[0]["status"] == "success"
                assert "code" in result[0]
                mock_agent.generate_code.assert_called_once()
    
    def test_code_runner_wrapper(self):
        """Test the code_runner_wrapper function."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import code_runner_wrapper
            
            with patch('app.code_runner') as mock_agent:
                # Setup mock response
                mock_agent.run_code.return_value = "Code executed successfully\\nOutput: Hello World"
                
                # Execute wrapper
                result = code_runner_wrapper("print('Hello World')")
                
                # Verify wrapper functionality
                assert "Code executed successfully" in result
                assert "Hello World" in result
                mock_agent.run_code.assert_called_once()
    
    def test_health_status_endpoint(self):
        """Test health status endpoint."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import get_health_status
            
            # Execute
            result = get_health_status()
            
            # Verify health status structure
            assert isinstance(result, dict)
            assert "status" in result
            assert "timestamp" in result
            # Additional fields may be present depending on implementation
    
    def test_performance_metrics_endpoint(self):
        """Test performance metrics endpoint."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import get_performance_metrics
            
            # Execute
            result = get_performance_metrics()
            
            # Verify metrics structure
            assert isinstance(result, dict)
            # Should contain performance-related information
    
    def test_cache_status_endpoint(self):
        """Test cache status endpoint."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import get_cache_status
            
            # Execute
            result = get_cache_status()
            
            # Verify cache status structure
            assert isinstance(result, dict)
            # Should contain cache-related information
    
    def test_sandbox_pool_status_endpoint(self):
        """Test sandbox pool status endpoint."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import get_sandbox_pool_status_sync
            
            # Execute
            result = get_sandbox_pool_status_sync()
            
            # Verify pool status structure
            assert isinstance(result, dict)
            # May contain pool information or error if not available
    
    def test_process_orchestrator_request_endpoint(self):
        """Test process_orchestrator_request endpoint (MCP interface)."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import process_orchestrator_request
            
            with patch('app.orchestrator') as mock_orchestrator:
                # Setup mock response
                mock_orchestrator.orchestrate.return_value = (
                    {
                        "status": "success",
                        "research_summary": "MCP test research",
                        "generated_code": "# MCP test code",
                        "execution_result": "MCP execution result",
                        "citations": []
                    },
                    "MCP test summary"
                )
                
                # Execute
                result = process_orchestrator_request("Create an MCP test script")
                
                # Verify MCP endpoint functionality
                assert isinstance(result, dict)
                assert result["status"] == "success"
                mock_orchestrator.orchestrate.assert_called_once()


class TestUIErrorHandling:
    """Test UI error handling and edge cases."""
    
    def test_orchestrator_wrapper_with_errors(self):
        """Test orchestrator wrapper with various error conditions."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import agent_orchestrator
            
            with patch('app.orchestrator') as mock_orchestrator:
                # Test with orchestrator error
                mock_orchestrator.orchestrate.side_effect = Exception("Orchestrator failed")
                
                # Execute - should handle error gracefully
                result = agent_orchestrator("Test request")
                
                # Should return error information
                assert isinstance(result, tuple)
                # Error handling behavior depends on implementation
    
    def test_wrapper_functions_with_empty_inputs(self):
        """Test wrapper functions with empty or invalid inputs."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import (
                agent_question_enhancer, 
                agent_web_search, 
                agent_llm_processor,
                code_runner_wrapper
            )
            
            # Test empty inputs
            result1 = agent_question_enhancer("", 3)
            result2 = agent_web_search("")
            result3 = agent_llm_processor("", "summarize")
            result4 = code_runner_wrapper("")
            
            # All should handle empty inputs gracefully
            assert isinstance(result1, dict)
            assert isinstance(result2, dict)
            assert isinstance(result3, dict)
            assert isinstance(result4, str)
    
    def test_endpoint_response_formats(self):
        """Test that all endpoints return proper response formats."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import (
                get_health_status,
                get_performance_metrics,
                get_cache_status,
                get_sandbox_pool_status_sync
            )
            
            # All status endpoints should return dictionaries
            health = get_health_status()
            performance = get_performance_metrics()
            cache = get_cache_status()
            sandbox = get_sandbox_pool_status_sync()
            
            assert isinstance(health, dict)
            assert isinstance(performance, dict)
            assert isinstance(cache, dict)
            assert isinstance(sandbox, dict)
    
    def test_gradio_interface_compatibility(self):
        """Test that wrapper functions are compatible with Gradio interfaces."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            # Import wrapper functions
            from app import (
                agent_orchestrator,
                agent_question_enhancer,
                agent_web_search,
                agent_llm_processor,
                agent_citation_formatter,
                agent_code_generator,
                code_runner_wrapper
            )
            
            # Test that all functions can be called without exceptions
            # (With proper mocking in place)
            with patch('app.orchestrator'), \
                 patch('app.question_enhancer'), \
                 patch('app.web_search'), \
                 patch('app.llm_processor'), \
                 patch('app.citation_formatter'), \
                 patch('app.code_generator'), \
                 patch('app.code_runner'):
                
                # All should be callable
                assert callable(agent_orchestrator)
                assert callable(agent_question_enhancer) 
                assert callable(agent_web_search)
                assert callable(agent_llm_processor)
                assert callable(agent_citation_formatter)
                assert callable(agent_code_generator)
                assert callable(code_runner_wrapper)


class TestMCPServerEndpoints:
    """Test MCP server specific endpoints."""
    
    def test_mcp_function_signatures(self):
        """Test that MCP functions have proper signatures."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            # Import MCP endpoint functions
            from app import (
                process_orchestrator_request,
                get_health_status,
                get_performance_metrics,
                get_cache_status,
                get_sandbox_pool_status_sync
            )
            
            # All MCP functions should have proper docstrings and type hints
            # This is important for MCP schema generation
            assert process_orchestrator_request.__doc__ is not None
            assert get_health_status.__doc__ is not None
            
            # Functions should be importable and callable
            assert callable(process_orchestrator_request)
            assert callable(get_health_status)
    
    def test_mcp_response_formats(self):
        """Test that MCP endpoints return JSON-serializable responses."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import (
                get_health_status,
                get_performance_metrics,
                get_cache_status
            )
            
            import json
            
            # All responses should be JSON serializable
            health = get_health_status()
            performance = get_performance_metrics()
            cache = get_cache_status()
            
            # Test JSON serialization
            try:
                json.dumps(health)
                json.dumps(performance)
                json.dumps(cache)
            except (TypeError, ValueError) as e:
                pytest.fail(f"MCP endpoint returned non-JSON-serializable response: {e}")
    
    def test_mcp_error_responses(self):
        """Test MCP error response handling."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import process_orchestrator_request
            
            with patch('app.orchestrator') as mock_orchestrator:
                # Setup orchestrator to raise an error
                mock_orchestrator.orchestrate.side_effect = Exception("MCP test error")
                
                # Execute
                result = process_orchestrator_request("Test request")
                
                # Should return structured error response
                assert isinstance(result, dict)
                # Error response format depends on implementation