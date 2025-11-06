"""CitationFormatterAgent - Formats citations from URLs in text."""

from typing import Dict, Any
from mcp_hub.exceptions import ValidationError
from mcp_hub.utils import validate_non_empty_string, extract_urls_from_text, create_apa_citation
from mcp_hub.logging_config import logger

# Import decorators with graceful fallback
try:
    from mcp_hub.performance_monitoring import track_performance
except ImportError:
    # Fallback decorator if advanced features not available
    def track_performance(operation_name: str = None):
        def decorator(func):
            return func
        return decorator


class CitationFormatterAgent:
    """
    Agent responsible for formatting citations from text content.

    This agent extracts URLs from text blocks and produces properly formatted
    APA-style citations. It handles the automated creation of academic references
    from web sources found in research content.
    """

    @track_performance(operation_name="citation_formatting")
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
