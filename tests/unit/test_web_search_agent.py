"""Unit tests for WebSearchAgent."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestWebSearchAgent:
    """Test WebSearchAgent functionality."""
    
    def test_web_search_agent_instantiation(self):
        """Test WebSearchAgent can be instantiated with proper mocking."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient') as mock_tavily:
            
            from app import WebSearchAgent
            
            # Test instantiation
            agent = WebSearchAgent()
            assert agent is not None
            # TavilyClient may be called multiple times due to global instances
            assert mock_tavily.called
    
    def test_search_basic_functionality(self):
        """Test basic search functionality."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient') as mock_tavily_class:
            
            from app import WebSearchAgent
            
            # Setup mock Tavily client
            mock_client = Mock()
            mock_tavily_class.return_value = mock_client
            
            search_results = {
                "results": [
                    {
                        "title": "Python Data Analysis",
                        "url": "https://example.com/data",
                        "content": "Learn data analysis with Python",
                        "score": 0.95
                    }
                ],
                "answer": "Python is great for data analysis"
            }
            mock_client.search.return_value = search_results
            
            # Execute
            agent = WebSearchAgent()
            result = agent.search("Python data analysis")
            
            # Verify
            assert result["status"] == "success"
            assert "results" in result
            assert "tavily_answer" in result
            assert len(result["results"]) == 1
            assert result["results"][0]["title"] == "Python Data Analysis"
            mock_client.search.assert_called_once()
    
    def test_search_with_filters(self):
        """Test search with topic filter and result limits."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient') as mock_tavily_class:
            
            from app import WebSearchAgent
            
            # Setup mock
            mock_client = Mock()
            mock_tavily_class.return_value = mock_client
            mock_client.search.return_value = {
                "results": [{"title": f"Result {i}", "url": f"https://example.com/{i}"}
                           for i in range(10)],
                "answer": "Search answer"
            }
            
            # Execute with filters
            agent = WebSearchAgent()
            result = agent.search("test query", topic="programming", max_results=3)
            
            # Verify search parameters
            call_args = mock_client.search.call_args[1]
            assert "query" in call_args
            assert call_args["max_results"] == 3
            assert call_args["topic"] == "programming"
    
    def test_search_error_handling(self):
        """Test error handling in search."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient') as mock_tavily_class:
            
            from app import WebSearchAgent
            from mcp_hub.exceptions import APIError
            
            # Setup error scenario
            mock_client = Mock()
            mock_tavily_class.return_value = mock_client
            mock_client.search.side_effect = Exception("API Error")
            
            # Execute
            agent = WebSearchAgent()
            result = agent.search("test query")
            
            # Verify error handling
            assert result["status"] == "error"
            assert "error" in result
            assert "API Error" in result["error"]
    
    def test_search_empty_query(self):
        """Test search with empty query."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient') as mock_tavily_class:
            
            from app import WebSearchAgent
            
            mock_client = Mock()
            mock_tavily_class.return_value = mock_client
            
            # Execute with empty query
            agent = WebSearchAgent()
            result = agent.search("")
            
            # Verify validation
            assert result["status"] == "error"
            assert "error" in result
    
    def test_search_no_results(self):
        """Test search with no results returned."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient') as mock_tavily_class:
            
            from app import WebSearchAgent
            
            # Setup empty results
            mock_client = Mock()
            mock_tavily_class.return_value = mock_client
            mock_client.search.return_value = {
                "results": [],
                "answer": ""
            }
            
            # Execute
            agent = WebSearchAgent()
            result = agent.search("obscure query")
            
            # Verify handling of empty results
            assert result["status"] == "success"
            assert result["results"] == []
            assert "tavily_answer" in result
    
    def test_search_api_config_validation(self):
        """Test that API configuration is validated on instantiation."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.api_config') as mock_config:
            
            from app import WebSearchAgent
            from mcp_hub.exceptions import APIError
            
            # Test missing API key
            mock_config.tavily_api_key = ""
            
            with pytest.raises(APIError):
                WebSearchAgent()
    
    def test_search_result_format(self):
        """Test that search results are properly formatted."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient') as mock_tavily_class:
            
            from app import WebSearchAgent
            
            # Setup mock with various result formats
            mock_client = Mock()
            mock_tavily_class.return_value = mock_client
            mock_client.search.return_value = {
                "results": [
                    {
                        "title": "Complete Result",
                        "url": "https://example.com/complete",
                        "content": "Full content",
                        "score": 0.95
                    },
                    {
                        "title": "Partial Result",
                        "url": "https://example.com/partial"
                        # Missing content and score
                    }
                ],
                "answer": "Search summary"
            }
            
            # Execute
            agent = WebSearchAgent()
            result = agent.search("test query")
            
            # Verify result format consistency
            assert result["status"] == "success"
            assert len(result["results"]) == 2
            
            # Check that all results have required fields
            for res in result["results"]:
                assert "title" in res
                assert "url" in res
                # Content and score are optional
    
    def test_search_logging(self):
        """Test that search operations are properly logged."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient') as mock_tavily_class, \
             patch('app.logger') as mock_logger:
            
            from app import WebSearchAgent
            
            mock_client = Mock()
            mock_tavily_class.return_value = mock_client
            mock_client.search.return_value = {"results": [], "answer": ""}
            
            # Execute
            agent = WebSearchAgent()
            agent.search("test query")
            
            # Verify logging occurred
            assert mock_logger.info.call_count >= 1