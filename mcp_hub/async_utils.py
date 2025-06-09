"""Async utilities for improved performance in concurrent operations."""

import asyncio
import aiohttp
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from .config import api_config, app_config
from .exceptions import APIError
from .logging_config import logger

class AsyncWebSearchAgent:
    """Async version of web search for concurrent operations."""
    
    def __init__(self):
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def search_multiple_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Search multiple queries concurrently."""
        if not self.session:
            raise APIError("AsyncWebSearch", "Session not initialized. Use as async context manager.")
        
        logger.info(f"Starting concurrent search for {len(queries)} queries")
        
        # Create tasks for concurrent execution
        tasks = [self._search_single_query(query) for query in queries]
        
        # Execute all searches concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Search failed for query {i}: {str(result)}")
                processed_results.append({
                    "error": str(result),
                    "query": queries[i],
                    "results": []
                })
            else:
                processed_results.append(result)
        
        logger.info(f"Completed concurrent searches: {len([r for r in processed_results if not r.get('error')])} successful")
        return processed_results
    
    async def _search_single_query(self, query: str) -> Dict[str, Any]:
        """Search a single query using Tavily API."""
        try:
            # In a real implementation, you'd make async HTTP calls to Tavily
            # For now, we'll use the sync version in a thread pool
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_config.tavily_api_key)
            
            # Run sync operation in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                response = await loop.run_in_executor(
                    executor,
                    lambda: client.search(
                        query=query,
                        search_depth="basic",
                        max_results=app_config.max_search_results,
                        include_answer=True
                    )
                )
            
            return {
                "query": response.get("query", query),
                "tavily_answer": response.get("answer"),
                "results": response.get("results", []),
                "data_source": "Tavily Search API (Async)",
            }
            
        except Exception as e:
            raise APIError("Tavily", f"Async search failed: {str(e)}")

async def process_subquestions_concurrently(sub_questions: List[str]) -> List[Dict[str, Any]]:
    """Process multiple sub-questions concurrently for better performance."""
    logger.info(f"Processing {len(sub_questions)} sub-questions concurrently")
    
    async with AsyncWebSearchAgent() as async_searcher:
        # Execute all searches concurrently
        search_results = await async_searcher.search_multiple_queries(sub_questions)
        
        return search_results
