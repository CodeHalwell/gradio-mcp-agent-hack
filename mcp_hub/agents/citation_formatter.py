"""
Citation Formatter Agent module.

This module contains the CitationFormatterAgent class which is responsible for 
formatting citations from text content.
"""

from typing import Dict, Any

from ..exceptions import ValidationError
from ..utils import validate_non_empty_string, extract_urls_from_text, create_apa_citation
from ..logging_config import logger

# Import advanced features with graceful fallback
try:
    from ..performance_monitoring import track_performance
except ImportError:
    # Create dummy decorators for backward compatibility
    def track_performance(operation_name: str = None):
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


class CitationFormatterAgent:
    """
    Agent responsible for formatting citations from text content.

    This agent extracts URLs from text blocks and produces properly formatted
    APA-style citations. It handles the automated creation of academic references
    from web sources found in research content.
    """
    
    @with_performance_tracking("citation_formatting")
    def format_citations(self, text_block: str) -> Dict[str, Any]:
        """
        Extract URLs from text and produce APA-style citations.

        Analyzes the provided text block to identify URLs and automatically
        generates properly formatted academic citations following APA style
        guidelines for web sources.

        Args:
            text_block (str): The text content containing URLs to be cited

        Returns:
            Dict[str, Any]: A dictionary containing formatted citations array
                           or error information if extraction fails
        """
        try:
            validate_non_empty_string(text_block, "Text block")
            logger.info("Formatting citations from text block")
            
            urls = extract_urls_from_text(text_block)
            if not urls:
                return {"error": "No URLs found to cite.", "formatted_citations": []}
            
            citations = []
            for url in urls:
                citation = create_apa_citation(url)
                citations.append(citation)
            
            logger.info(f"Successfully formatted {len(citations)} citations")
            return {"formatted_citations": citations, "error": None}
            
        except ValidationError as e:
            logger.error(f"Citation formatting validation failed: {str(e)}")
            return {"error": str(e), "formatted_citations": []}
        except Exception as e:
            logger.error(f"Citation formatting failed: {str(e)}")
            return {"error": f"Unexpected error: {str(e)}", "formatted_citations": []}