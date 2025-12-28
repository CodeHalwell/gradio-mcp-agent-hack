"""Integration tests for individual agents.

These tests verify that each agent works correctly with real or mocked
dependencies, testing the integration between the agent and external services.
"""

import pytest
from unittest.mock import Mock, patch
from mcp_hub.agents import (
    QuestionEnhancerAgent,
    WebSearchAgent,
    LLMProcessorAgent,
    CitationFormatterAgent,
    CodeGeneratorAgent,
    CodeRunnerAgent,
)


@pytest.mark.integration
class TestQuestionEnhancerIntegration:
    """Integration tests for QuestionEnhancerAgent."""

    def test_enhance_question_with_mock_llm(self):
        """Test question enhancement with mocked LLM."""
        # Mock LLM response
        mock_response = '''
        {
            "sub_questions": [
                "What is the syntax for Python list comprehensions?",
                "How do list comprehensions compare to traditional loops?",
                "What are common use cases for list comprehensions?"
            ]
        }
        '''

        with patch('mcp_hub.agents.question_enhancer.make_llm_completion') as mock_llm:
            mock_llm.return_value = mock_response

            agent = QuestionEnhancerAgent()
            result = agent.enhance_question(
                "How do I use list comprehensions in Python?",
                num_questions=3
            )

            # Verify structure
            assert 'sub_questions' in result
            assert isinstance(result['sub_questions'], list)
            assert len(result['sub_questions']) >= 1

            # Verify LLM was called
            mock_llm.assert_called_once()
            call_args = mock_llm.call_args
            assert 'messages' in call_args.kwargs

    def test_enhance_question_handles_invalid_json(self):
        """Test that agent handles invalid JSON from LLM gracefully."""
        with patch('mcp_hub.agents.question_enhancer.make_llm_completion') as mock_llm:
            mock_llm.return_value = "This is not JSON"

            agent = QuestionEnhancerAgent()
            result = agent.enhance_question("Test question", 3)

            # Should still return a valid structure
            assert 'sub_questions' in result
            assert isinstance(result['sub_questions'], list)

    def test_enhance_question_with_different_counts(self):
        """Test enhancing questions with different sub-question counts."""
        mock_response = '''
        {
            "sub_questions": [
                "Question 1?",
                "Question 2?",
                "Question 3?",
                "Question 4?",
                "Question 5?"
            ]
        }
        '''

        with patch('mcp_hub.agents.question_enhancer.make_llm_completion') as mock_llm:
            mock_llm.return_value = mock_response

            agent = QuestionEnhancerAgent()

            # Test with 5 questions
            result = agent.enhance_question("Test question", num_questions=5)
            assert len(result['sub_questions']) >= 1


@pytest.mark.integration
class TestWebSearchIntegration:
    """Integration tests for WebSearchAgent."""

    def test_search_with_mock_tavily(self):
        """Test web search with mocked Tavily client."""
        mock_results = {
            "results": [
                {
                    "title": "Python Documentation",
                    "url": "https://docs.python.org",
                    "content": "Official Python documentation",
                    "score": 0.95
                },
                {
                    "title": "Real Python Tutorial",
                    "url": "https://realpython.com",
                    "content": "Learn Python programming",
                    "score": 0.90
                }
            ],
            "query": "python programming tutorial",
            "answer": "Python is a high-level programming language..."
        }

        with patch('mcp_hub.agents.web_search.TavilyClient') as MockTavilyClient:
            mock_client = Mock()
            mock_client.search.return_value = mock_results
            MockTavilyClient.return_value = mock_client

            agent = WebSearchAgent()
            result = agent.search("python programming tutorial")

            # Verify structure
            assert 'results' in result
            assert 'query' in result
            assert isinstance(result['results'], list)
            assert len(result['results']) > 0

            # Verify first result structure
            first_result = result['results'][0]
            assert 'title' in first_result
            assert 'url' in first_result
            assert 'content' in first_result

    def test_search_handles_api_errors(self):
        """Test that search handles API errors gracefully."""
        with patch('mcp_hub.agents.web_search.TavilyClient') as MockTavilyClient:
            mock_client = Mock()
            mock_client.search.side_effect = Exception("API Error")
            MockTavilyClient.return_value = mock_client

            agent = WebSearchAgent()

            with pytest.raises(Exception):
                agent.search("test query")

    def test_search_with_empty_query(self):
        """Test search with empty query."""
        agent = WebSearchAgent()

        with pytest.raises((ValueError, Exception)):
            agent.search("")


@pytest.mark.integration
class TestLLMProcessorIntegration:
    """Integration tests for LLMProcessorAgent."""

    def test_summarize_with_mock_llm(self):
        """Test text summarization."""
        mock_summary = "This is a concise summary of the text."

        with patch('mcp_hub.agents.llm_processor.make_llm_completion') as mock_llm:
            mock_llm.return_value = mock_summary

            agent = LLMProcessorAgent()
            result = agent.process(
                text_input="Long article about Python programming...",
                task="summarize",
                context="Python programming"
            )

            # Verify structure
            assert 'result' in result
            assert 'task' in result
            assert result['task'] == 'summarize'
            assert isinstance(result['result'], str)

            # Verify LLM was called
            mock_llm.assert_called_once()

    def test_reason_with_mock_llm(self):
        """Test reasoning task."""
        mock_reasoning = "Based on the facts, we can conclude that..."

        with patch('mcp_hub.agents.llm_processor.make_llm_completion') as mock_llm:
            mock_llm.return_value = mock_reasoning

            agent = LLMProcessorAgent()
            result = agent.process(
                text_input="Fact 1. Fact 2. Fact 3.",
                task="reason",
                context="Draw conclusions"
            )

            assert result['task'] == 'reason'
            assert len(result['result']) > 0

    def test_extract_keywords_with_mock_llm(self):
        """Test keyword extraction."""
        mock_keywords = "python, programming, tutorial, learning, code"

        with patch('mcp_hub.agents.llm_processor.make_llm_completion') as mock_llm:
            mock_llm.return_value = mock_keywords

            agent = LLMProcessorAgent()
            result = agent.process(
                text_input="This is an article about Python programming...",
                task="extract_keywords"
            )

            assert result['task'] == 'extract_keywords'
            assert len(result['result']) > 0

    def test_invalid_task_raises_error(self):
        """Test that invalid task types raise errors."""
        agent = LLMProcessorAgent()

        with pytest.raises((ValueError, Exception)):
            agent.process(
                text_input="Some text",
                task="invalid_task"
            )


@pytest.mark.integration
class TestCitationFormatterIntegration:
    """Integration tests for CitationFormatterAgent."""

    def test_format_citation_basic(self):
        """Test basic citation formatting."""
        agent = CitationFormatterAgent()
        result = agent.format_citation("https://example.com/article")

        # Should return a string
        assert isinstance(result, str)
        assert len(result) > 0
        assert "example.com" in result.lower() or "Example" in result

    def test_format_citation_with_path(self):
        """Test citation with URL path."""
        agent = CitationFormatterAgent()
        result = agent.format_citation("https://docs.python.org/3/library/asyncio.html")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_citation_handles_invalid_url(self):
        """Test handling of invalid URLs."""
        agent = CitationFormatterAgent()

        # Should still return something (graceful degradation)
        result = agent.format_citation("not-a-valid-url")
        assert isinstance(result, str)


@pytest.mark.integration
class TestCodeGeneratorIntegration:
    """Integration tests for CodeGeneratorAgent."""

    def test_generate_code_with_mock_llm(self):
        """Test code generation with mocked LLM."""
        mock_code = '''
def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
        '''

        with patch('mcp_hub.agents.code_generator.make_llm_completion') as mock_llm:
            mock_llm.return_value = mock_code

            agent = CodeGeneratorAgent()
            result = agent.generate(
                user_request="Create a function to calculate fibonacci numbers",
                grounded_context="The function should be recursive"
            )

            # Verify structure
            assert 'code' in result
            assert 'validation_passed' in result
            assert 'imports' in result
            assert isinstance(result['code'], str)
            assert isinstance(result['validation_passed'], bool)

    def test_generate_code_rejects_dangerous_functions(self):
        """Test that dangerous functions are rejected."""
        dangerous_code = '''
import os
eval("malicious code")
exec("dangerous")
        '''

        with patch('mcp_hub.agents.code_generator.make_llm_completion') as mock_llm:
            mock_llm.return_value = dangerous_code

            agent = CodeGeneratorAgent()
            result = agent.generate(
                user_request="Test",
                grounded_context="Test"
            )

            # Should fail validation
            assert result['validation_passed'] is False
            assert len(result.get('disallowed_functions', [])) > 0

    def test_generate_code_allows_safe_code(self):
        """Test that safe code passes validation."""
        safe_code = '''
def add_numbers(a, b):
    """Add two numbers."""
    return a + b

result = add_numbers(5, 3)
print(f"Result: {result}")
        '''

        with patch('mcp_hub.agents.code_generator.make_llm_completion') as mock_llm:
            mock_llm.return_value = safe_code

            agent = CodeGeneratorAgent()
            result = agent.generate(
                user_request="Create a function to add numbers",
                grounded_context="Simple addition"
            )

            # Should pass validation
            assert result['validation_passed'] is True

    def test_generate_code_extracts_imports(self):
        """Test import extraction."""
        code_with_imports = '''
import json
import sys
from typing import List, Dict

def process_data(data: List[Dict]) -> str:
    return json.dumps(data)
        '''

        with patch('mcp_hub.agents.code_generator.make_llm_completion') as mock_llm:
            mock_llm.return_value = code_with_imports

            agent = CodeGeneratorAgent()
            result = agent.generate(
                user_request="Test",
                grounded_context="Test"
            )

            # Should extract imports
            assert 'imports' in result
            assert isinstance(result['imports'], list)


@pytest.mark.integration
class TestCodeRunnerIntegration:
    """Integration tests for CodeRunnerAgent."""

    @pytest.mark.skip(reason="Requires Modal setup")
    def test_run_code_simple(self):
        """Test running simple code."""
        code = 'print("Hello, World!")'

        agent = CodeRunnerAgent()
        result = agent.run(code)

        # Should return output
        assert isinstance(result, str)
        assert "Hello, World!" in result

    @pytest.mark.skip(reason="Requires Modal setup")
    def test_run_code_with_calculation(self):
        """Test running code with calculations."""
        code = '''
result = 2 + 2
print(f"2 + 2 = {result}")
        '''

        agent = CodeRunnerAgent()
        result = agent.run(code)

        assert "4" in result

    @pytest.mark.skip(reason="Requires Modal setup")
    def test_run_code_with_error(self):
        """Test running code that raises an error."""
        code = '''
x = 1 / 0  # Division by zero
        '''

        agent = CodeRunnerAgent()
        result = agent.run(code)

        # Should capture error
        assert isinstance(result, str)
        assert "Error" in result or "Exception" in result or "ZeroDivisionError" in result

    def test_run_code_validates_input(self):
        """Test that code runner validates input."""
        agent = CodeRunnerAgent()

        # Empty code should raise error
        with pytest.raises((ValueError, Exception)):
            agent.run("")


@pytest.mark.integration
class TestAgentsWorkflowIntegration:
    """Integration tests for multi-agent workflows."""

    def test_question_to_search_workflow(self):
        """Test workflow from question enhancement to search."""
        # Mock LLM for question enhancement
        mock_questions = '''
        {
            "sub_questions": [
                "What is Python?",
                "How to install Python?"
            ]
        }
        '''

        # Mock search results
        mock_search = {
            "results": [
                {
                    "title": "Python",
                    "url": "https://python.org",
                    "content": "Python is a programming language",
                    "score": 0.9
                }
            ],
            "query": "What is Python?",
            "answer": "Python is a programming language"
        }

        with patch('mcp_hub.agents.question_enhancer.make_llm_completion') as mock_llm:
            with patch('mcp_hub.agents.web_search.TavilyClient') as MockTavily:
                mock_llm.return_value = mock_questions

                mock_client = Mock()
                mock_client.search.return_value = mock_search
                MockTavily.return_value = mock_client

                # Step 1: Enhance question
                enhancer = QuestionEnhancerAgent()
                enhanced = enhancer.enhance_question("Tell me about Python", 3)

                # Step 2: Search for first sub-question
                searcher = WebSearchAgent()
                results = searcher.search(enhanced['sub_questions'][0])

                # Verify workflow
                assert len(enhanced['sub_questions']) > 0
                assert 'results' in results
                assert len(results['results']) > 0

    def test_search_to_llm_to_code_workflow(self):
        """Test workflow from search through LLM processing to code generation."""
        # Mock search
        mock_search = {
            "results": [
                {
                    "title": "Python Async",
                    "url": "https://docs.python.org/3/library/asyncio.html",
                    "content": "asyncio is a library for async programming",
                    "score": 0.95
                }
            ],
            "query": "Python async programming",
            "answer": "asyncio provides async/await syntax"
        }

        # Mock LLM summarization
        mock_summary = "asyncio is Python's library for asynchronous programming."

        # Mock code generation
        mock_code = '''
import asyncio

async def main():
    print("Async function")
    await asyncio.sleep(1)

asyncio.run(main())
        '''

        with patch('mcp_hub.agents.web_search.TavilyClient') as MockTavily:
            with patch('mcp_hub.agents.llm_processor.make_llm_completion') as mock_llm_process:
                with patch('mcp_hub.agents.code_generator.make_llm_completion') as mock_llm_code:
                    # Setup mocks
                    mock_client = Mock()
                    mock_client.search.return_value = mock_search
                    MockTavily.return_value = mock_client
                    mock_llm_process.return_value = mock_summary
                    mock_llm_code.return_value = mock_code

                    # Step 1: Search
                    searcher = WebSearchAgent()
                    search_results = searcher.search("Python async programming")

                    # Step 2: Summarize
                    processor = LLMProcessorAgent()
                    summary = processor.process(
                        text_input=search_results['results'][0]['content'],
                        task="summarize"
                    )

                    # Step 3: Generate code
                    generator = CodeGeneratorAgent()
                    code_result = generator.generate(
                        user_request="Create an async function",
                        grounded_context=summary['result']
                    )

                    # Verify workflow
                    assert len(search_results['results']) > 0
                    assert 'result' in summary
                    assert 'code' in code_result
                    assert code_result['validation_passed'] is True


@pytest.mark.integration
class TestAgentsErrorHandling:
    """Integration tests for error handling across agents."""

    def test_agents_handle_network_errors(self):
        """Test that agents handle network errors gracefully."""
        with patch('mcp_hub.agents.web_search.TavilyClient') as MockTavily:
            mock_client = Mock()
            mock_client.search.side_effect = ConnectionError("Network error")
            MockTavily.return_value = mock_client

            agent = WebSearchAgent()

            with pytest.raises((ConnectionError, Exception)):
                agent.search("test query")

    def test_agents_handle_timeout_errors(self):
        """Test that agents handle timeout errors."""
        with patch('mcp_hub.agents.question_enhancer.make_llm_completion') as mock_llm:
            mock_llm.side_effect = TimeoutError("Request timeout")

            agent = QuestionEnhancerAgent()

            with pytest.raises((TimeoutError, Exception)):
                agent.enhance_question("test question", 3)

    def test_agents_handle_validation_errors(self):
        """Test that agents handle validation errors."""
        agent = CodeGeneratorAgent()

        # Very short request should fail validation
        with pytest.raises((ValueError, Exception)):
            agent.generate(
                user_request="x",  # Too short
                grounded_context="y"
            )
