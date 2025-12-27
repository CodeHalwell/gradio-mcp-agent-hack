"""Web Search Agent for performing internet searches using Tavily API."""
import aiohttp
from typing import Dict, Any

from tavily import TavilyClient

from ..config import api_config, app_config
from ..exceptions import APIError, ValidationError
from ..logging_config import logger
from ..utils import validate_non_empty_string
from ..decorators import with_performance_tracking, rate_limited, circuit_protected, cached


class WebSearchAgent:
    """
    Agent responsible for performing web searches using the Tavily API.

    This agent handles web search operations to gather information from the internet.
    It provides both synchronous and asynchronous search capabilities with configurable
    result limits and search depth. Results include summaries, URLs, and content snippets.
    """
    
    def __init__(self):
        if not api_config.tavily_api_key:
            raise APIError("Tavily", "API key not configured")
        self.client = TavilyClient(api_key=api_config.tavily_api_key)
    
    @with_performance_tracking("web_search")
    @rate_limited("tavily")
    @circuit_protected("tavily")
    @cached(ttl=600)  # Cache for 10 minutes
    def search(self, query: str) -> Dict[str, Any]:
        """
        Perform a web search using the Tavily API to gather internet information.

        Executes a synchronous web search with the specified query and returns
        structured results including search summaries, URLs, and content snippets.
        Results are cached for performance optimization.

        Args:
            query (str): The search query string to look up on the web

        Returns:
            Dict[str, Any]: A dictionary containing search results, summaries, and metadata
                           or error information if the search fails
        """
        try:
            validate_non_empty_string(query, "Search query")
            logger.info(f"Performing web search: {query}")
            
            response = self.client.search(
                query=query,
                search_depth="basic",
                max_results=app_config.max_search_results,
                include_answer=True
            )
            
            logger.info(f"Search completed, found {len(response.get('results', []))} results")
            return {
                "query": response.get("query", query),
                "tavily_answer": response.get("answer"),
                "results": response.get("results", []),
                "data_source": "Tavily Search API",
            }
            
        except ValidationError as e:
            logger.error(f"Web search validation failed: {str(e)}")
            return {"error": str(e), "query": query, "results": []}
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            return {"error": f"Tavily API Error: {str(e)}", "query": query, "results": []}
    
    @with_performance_tracking("async_web_search")
    @rate_limited("tavily")
    @circuit_protected("tavily")
    async def search_async(self, query: str) -> Dict[str, Any]:
        """
        Perform an asynchronous web search using aiohttp for better performance.

        Executes an async web search with the specified query using direct HTTP calls
        to the Tavily API. Falls back to synchronous search if async fails.
        Provides better performance for concurrent operations.

        Args:
            query (str): The search query string to look up on the web

        Returns:
            Dict[str, Any]: A dictionary containing search results, summaries, and metadata
                           or falls back to synchronous search on error
        """
        try:
            validate_non_empty_string(query, "Search query")
            logger.info(f"Performing async web search: {query}")
            
            # Use async HTTP client for better performance
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {api_config.tavily_api_key}',
                    'Content-Type': 'application/json'
                }
                
                payload = {
                    'query': query,
                    'search_depth': 'basic',
                    'max_results': app_config.max_search_results,
                    'include_answer': True
                }
                
                async with session.post(
                    'https://api.tavily.com/search',
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Async search completed, found {len(data.get('results', []))} results")
                        return {
                            "query": data.get("query", query),
                            "tavily_answer": data.get("answer"),
                            "results": data.get("results", []),
                            "data_source": "Tavily Search API (Async)",
                        }
                    else:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")
            
        except ValidationError as e:
            logger.error(f"Async web search validation failed: {str(e)}")
            return {"error": str(e), "query": query, "results": []}
        except Exception as e:
            logger.error(f"Async web search failed: {str(e)}")
            # Fallback to sync version on error
            logger.info("Falling back to synchronous search")
            return self.search(query)
