"""Citation Formatter Agent for formatting references from text content."""
from typing import Dict, Any

from ..exceptions import ValidationError
from ..logging_config import logger
from ..utils import validate_non_empty_string, extract_urls_from_text, create_apa_citation
from ..decorators import with_performance_tracking


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
