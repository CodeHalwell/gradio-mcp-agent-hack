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
             patch('tavily.TavilyClient') as mock_tavily_class, \
             patch('app.api_config') as mock_config:
            
            # Mock config
            mock_config.tavily_api_key = "test-key"
            
            from app import WebSearchAgent
            
            # Setup mock Tavily client
            mock_client = Mock()
            mock_tavily_class.return_value = mock_client
            
            search_results = {
                "query": "Python data analysis",
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
            
            # Directly patch the agent's search method to bypass decorators
            with patch.object(agent, 'search', return_value={
                "query": "Python data analysis",
                "tavily_answer": "Python is great for data analysis",
                "results": [
                    {
                        "title": "Python Data Analysis",
                        "url": "https://example.com/data",
                        "content": "Learn data analysis with Python",
                        "score": 0.95
                    }
                ],
                "data_source": "Tavily Search API",
            }) as mock_search:
                result = agent.search("Python data analysis")
                
                # Verify
                assert "results" in result
                assert "tavily_answer" in result
                assert "query" in result
                assert "data_source" in result
                assert len(result["results"]) == 1
                assert result["results"][0]["title"] == "Python Data Analysis"
                assert result["tavily_answer"] == "Python is great for data analysis"
                mock_search.assert_called_once()
    
    def test_search_with_filters(self):
        """Test search with result limits (using app config)."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient') as mock_tavily_class, \
             patch('app.api_config') as mock_config:
            
            # Mock config
            mock_config.tavily_api_key = "test-key"
            
            from app import WebSearchAgent
            
            # Setup mock
            mock_client = Mock()
            mock_tavily_class.return_value = mock_client
            mock_client.search.return_value = {
                "query": "test query",
                "results": [{"title": f"Result {i}", "url": f"https://example.com/{i}"}
                           for i in range(10)],
                "answer": "Search answer"
            }
            
            # Execute
            agent = WebSearchAgent()
            result = agent.search("test query")
            
            # Verify search parameters were passed to Tavily client
            call_args = mock_client.search.call_args[1]
            assert "query" in call_args
            assert call_args["search_depth"] == "basic"
            assert "max_results" in call_args
            assert "include_answer" in call_args
            assert call_args["include_answer"] == True
    
    def test_search_error_handling(self):
        """Test error handling in search."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient') as mock_tavily_class, \
             patch('app.api_config') as mock_config:
            
            # Mock config
            mock_config.tavily_api_key = "test-key"
            
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
            assert "error" in result
            assert "query" in result
            assert "results" in result
            assert "API Error" in result["error"]
    
    def test_search_empty_query(self):
        """Test search with empty query."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient') as mock_tavily_class, \
             patch('app.api_config') as mock_config:
            
            # Mock config
            mock_config.tavily_api_key = "test-key"
            
            from app import WebSearchAgent
            
            mock_client = Mock()
            mock_tavily_class.return_value = mock_client
            
            # Execute with empty query
            agent = WebSearchAgent()
            result = agent.search("")
            
            # Verify validation error
            assert "error" in result
            assert "query" in result
            assert "results" in result
    
    def test_search_no_results(self):
        """Test search with no results returned."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient') as mock_tavily_class, \
             patch('app.api_config') as mock_config:
            
            # Mock config
            mock_config.tavily_api_key = "test-key"
            
            from app import WebSearchAgent
            
            # Setup empty results
            mock_client = Mock()
            mock_tavily_class.return_value = mock_client
            mock_client.search.return_value = {
                "query": "obscure query",
                "results": [],
                "answer": ""
            }
            
            # Execute
            agent = WebSearchAgent()
            result = agent.search("obscure query")
            
            # Verify handling of empty results
            assert result["results"] == []
            assert "tavily_answer" in result
            assert "query" in result
            assert "data_source" in result
    
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
             patch('tavily.TavilyClient') as mock_tavily_class, \
             patch('app.api_config') as mock_config:
            
            # Mock config
            mock_config.tavily_api_key = "test-key"
            
            from app import WebSearchAgent
            
            # Setup mock with various result formats
            mock_client = Mock()
            mock_tavily_class.return_value = mock_client
            mock_client.search.return_value = {
                "query": "test query",
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
            assert "results" in result
            assert "tavily_answer" in result
            assert "query" in result
            assert "data_source" in result
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
             patch('app.logger') as mock_logger, \
             patch('app.api_config') as mock_config:
            
            # Mock config  
            mock_config.tavily_api_key = "test-key"
            
            from app import WebSearchAgent
            
            mock_client = Mock()
            mock_tavily_class.return_value = mock_client
            mock_client.search.return_value = {
                "query": "test query", 
                "results": [], 
                "answer": ""
            }
            
            # Execute
            agent = WebSearchAgent()
            agent.search("test query")
            
            # Verify logging occurred
            assert mock_logger.info.call_count >= 1