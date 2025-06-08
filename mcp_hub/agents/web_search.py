"""
Web Search Agent module.

This module contains the WebSearchAgent class which is responsible for 
performing web searches using the Tavily API.
"""

import aiohttp
from typing import Dict, Any

from tavily import TavilyClient

from ..config import api_config, app_config
from ..exceptions import ValidationError, APIError
from ..utils import validate_non_empty_string
from ..logging_config import logger

# Import advanced features with graceful fallback
try:
    from ..performance_monitoring import track_performance
    from ..cache_utils import cached
    from ..reliability_utils import rate_limited, circuit_protected
except ImportError:
    # Create dummy decorators for backward compatibility
    def track_performance(operation_name: str = None):
        def decorator(func): 
            return func
        return decorator
    
    def rate_limited(service: str = "default", timeout: float = 10.0):
        def decorator(func): 
            return func
        return decorator
    
    def circuit_protected(service: str = "default"):
        def decorator(func): 
            return func
        return decorator
    
    def cached(ttl: int = 300):
        def decorator(func): 
            return func
        return decorator

# Performance tracking wrapper (from app.py - needed for async methods)
import time
import asyncio
from functools import wraps


def with_performance_tracking(operation_name: str):
    """
    Add performance tracking and metrics collection to any function (sync or async).
    """
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                    raise
                finally:
                    end_time = time.time()
                    duration = end_time - start_time
                    logger.info(f"Operation {operation_name} completed in {duration:.3f}s, success: {success}")
                return result
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                    raise
                finally:
                    end_time = time.time()
                    duration = end_time - start_time
                    logger.info(f"Operation {operation_name} completed in {duration:.3f}s, success: {success}")
                return result
            return sync_wrapper
    return decorator


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