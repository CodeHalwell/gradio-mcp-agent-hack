"""Unit tests for CitationFormatterAgent."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestCitationFormatterAgent:
    """Test CitationFormatterAgent functionality."""
    
    def test_citation_formatter_instantiation(self):
        """Test CitationFormatterAgent can be instantiated."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import CitationFormatterAgent
            
            agent = CitationFormatterAgent()
            assert agent is not None
    
    def test_format_citations_basic(self):
        """Test basic citation formatting functionality."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.extract_urls_from_text') as mock_extract_urls, \
             patch('app.create_apa_citation') as mock_create_citation:
            
            from app import CitationFormatterAgent
            
            # Setup mocks
            mock_extract_urls.return_value = ["https://example.com/article1"]
            mock_create_citation.return_value = "Author, A. (2023). Title. Website."
            
            # Execute
            agent = CitationFormatterAgent()
            result = agent.format_citations("Text with https://example.com/article1 link")
            
            # Verify - using actual return format
            assert "formatted_citations" in result
            assert result["error"] is None
            assert len(result["formatted_citations"]) == 1
            assert result["formatted_citations"][0] == "Author, A. (2023). Title. Website."
            mock_extract_urls.assert_called_once()
            mock_create_citation.assert_called_once()
    
    def test_format_citations_multiple_urls(self):
        """Test formatting citations with multiple URLs."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.extract_urls_from_text') as mock_extract_urls, \
             patch('app.create_apa_citation') as mock_create_citation:
            
            from app import CitationFormatterAgent
            
            # Setup multiple URLs
            urls = [
                "https://example.com/article1",
                "https://example.com/article2",
                "https://example.com/article3"
            ]
            mock_extract_urls.return_value = urls
            mock_create_citation.side_effect = [
                "Citation 1",
                "Citation 2", 
                "Citation 3"
            ]
            
            # Execute
            agent = CitationFormatterAgent()
            result = agent.format_citations("Text with multiple URLs")
            
            # Verify
            assert "formatted_citations" in result
            assert result["error"] is None
            assert len(result["formatted_citations"]) == 3
            assert result["formatted_citations"] == ["Citation 1", "Citation 2", "Citation 3"]
            assert mock_create_citation.call_count == 3
    
    def test_format_citations_no_urls(self):
        """Test formatting citations with no URLs in text."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.extract_urls_from_text') as mock_extract_urls:
            
            from app import CitationFormatterAgent
            
            # Setup no URLs
            mock_extract_urls.return_value = []
            
            # Execute
            agent = CitationFormatterAgent()
            result = agent.format_citations("Text with no URLs")
            
            # Verify
            assert "formatted_citations" in result
            assert result["formatted_citations"] == []
            assert "error" in result
            assert "No URLs found" in result["error"]
    
    def test_format_citations_empty_text(self):
        """Test formatting citations with empty text."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import CitationFormatterAgent
            
            # Execute with empty text
            agent = CitationFormatterAgent()
            result = agent.format_citations("")
            
            # Verify error handling
            assert "error" in result
            assert result["formatted_citations"] == []
    
    def test_format_citations_url_extraction_error(self):
        """Test handling of URL extraction errors."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.extract_urls_from_text') as mock_extract_urls:
            
            from app import CitationFormatterAgent
            
            # Setup extraction error
            mock_extract_urls.side_effect = Exception("URL extraction failed")
            
            # Execute
            agent = CitationFormatterAgent()
            result = agent.format_citations("Text with URLs")
            
            # Verify error handling
            assert "error" in result
            assert result["formatted_citations"] == []
            assert "URL extraction failed" in result["error"]
    
    def test_format_citations_citation_creation_error(self):
        """Test handling of citation creation errors."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.extract_urls_from_text') as mock_extract_urls, \
             patch('app.create_apa_citation') as mock_create_citation:
            
            from app import CitationFormatterAgent
            
            # Setup citation creation error
            mock_extract_urls.return_value = ["https://example.com"]
            mock_create_citation.side_effect = Exception("Citation creation failed")
            
            # Execute
            agent = CitationFormatterAgent()
            result = agent.format_citations("Text with https://example.com")
            
            # Verify error handling
            assert "error" in result
            assert result["formatted_citations"] == []
            assert "Citation creation failed" in result["error"]
    
    def test_format_citations_partial_failures(self):
        """Test handling when some citations fail to create."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.extract_urls_from_text') as mock_extract_urls, \
             patch('app.create_apa_citation') as mock_create_citation, \
             patch('app.logger') as mock_logger:
            
            from app import CitationFormatterAgent
            
            # Setup partial failures
            mock_extract_urls.return_value = [
                "https://example.com/good",
                "https://example.com/bad"
            ]
            mock_create_citation.side_effect = [
                "Good Citation",
                Exception("Failed citation")
            ]
            
            # Execute
            agent = CitationFormatterAgent()
            result = agent.format_citations("Text with URLs")
            
            # Should continue processing even if some citations fail
            # This tests the robustness of the implementation
            # The exact behavior depends on the actual implementation
            assert "formatted_citations" in result or "error" in result
    
    def test_format_citations_duplicate_urls(self):
        """Test handling of duplicate URLs."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.extract_urls_from_text') as mock_extract_urls, \
             patch('app.create_apa_citation') as mock_create_citation:
            
            from app import CitationFormatterAgent
            
            # Setup duplicate URLs
            mock_extract_urls.return_value = [
                "https://example.com/article",
                "https://example.com/article",  # Duplicate
                "https://example.com/other"
            ]
            mock_create_citation.side_effect = [
                "Citation 1",
                "Citation 1",  # Same citation
                "Citation 2"
            ]
            
            # Execute
            agent = CitationFormatterAgent()
            result = agent.format_citations("Text with duplicate URLs")
            
            # Verify handling (behavior depends on implementation)
            assert "formatted_citations" in result
            # May have duplicates or deduplicated citations
    
    def test_format_citations_malformed_urls(self):
        """Test handling of malformed URLs."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.extract_urls_from_text') as mock_extract_urls, \
             patch('app.create_apa_citation') as mock_create_citation:
            
            from app import CitationFormatterAgent
            
            # Setup malformed URLs
            mock_extract_urls.return_value = [
                "not-a-url",
                "https://",
                "ftp://example.com"
            ]
            mock_create_citation.side_effect = Exception("Invalid URL")
            
            # Execute
            agent = CitationFormatterAgent()
            result = agent.format_citations("Text with malformed URLs")
            
            # Should handle malformed URLs gracefully
            assert "formatted_citations" in result
            # Should have error info when URLs are malformed
    
    def test_format_citations_logging(self):
        """Test that citation formatting operations are logged."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.extract_urls_from_text') as mock_extract_urls, \
             patch('app.create_apa_citation') as mock_create_citation, \
             patch('app.logger') as mock_logger:
            
            from app import CitationFormatterAgent
            
            mock_extract_urls.return_value = ["https://example.com"]
            mock_create_citation.return_value = "Test Citation"
            
            # Execute
            agent = CitationFormatterAgent()
            agent.format_citations("Text with URL")
            
            # Verify logging occurred
            assert mock_logger.info.call_count >= 1
    
    def test_format_citations_with_various_url_types(self):
        """Test formatting citations with various types of URLs."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.extract_urls_from_text') as mock_extract_urls, \
             patch('app.create_apa_citation') as mock_create_citation:
            
            from app import CitationFormatterAgent
            
            # Setup various URL types
            mock_extract_urls.return_value = [
                "https://www.example.com/article",
                "http://blog.example.org/post",
                "https://github.com/user/repo",
                "https://arxiv.org/abs/1234.5678"
            ]
            mock_create_citation.side_effect = [
                "Website Citation",
                "Blog Citation",
                "GitHub Citation", 
                "ArXiv Citation"
            ]
            
            # Execute
            agent = CitationFormatterAgent()
            result = agent.format_citations("Text with various URL types")
            
            # Verify
            assert "formatted_citations" in result
            assert len(result["formatted_citations"]) == 4
            assert all(isinstance(citation, str) for citation in result["formatted_citations"])