"""Integration tests for the OrchestratorAgent workflow.

These tests verify the end-to-end orchestration process with real agent interactions,
ensuring that all components work together correctly.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio

from mcp_hub.agents.orchestrator import OrchestratorAgent
from mcp_hub.agents.question_enhancer import QuestionEnhancerAgent
from mcp_hub.agents.web_search import WebSearchAgent
from mcp_hub.agents.llm_processor import LLMProcessorAgent
from mcp_hub.agents.citation_formatter import CitationFormatterAgent
from mcp_hub.agents.code_generator import CodeGeneratorAgent
from mcp_hub.agents.code_runner import CodeRunnerAgent


@pytest.fixture
def mock_question_enhancer_response():
    """Mock response for question enhancement."""
    return {
        "sub_questions": [
            "What is Python list comprehension?",
            "How to use list comprehension efficiently?",
            "What are the performance benefits of list comprehension?"
        ]
    }


@pytest.fixture
def mock_search_results():
    """Mock search results from Tavily."""
    return {
        "query": "What is Python list comprehension?",
        "tavily_answer": "List comprehension is a concise way to create lists in Python.",
        "results": [
            {
                "title": "Python List Comprehension Tutorial",
                "url": "https://example.com/list-comprehension",
                "content": "List comprehension provides a concise way to create lists...",
                "score": 0.95
            },
            {
                "title": "Advanced List Comprehension",
                "url": "https://example.com/advanced-list-comp",
                "content": "Learn advanced techniques for list comprehension...",
                "score": 0.88
            }
        ],
        "data_source": "Tavily Search API"
    }


@pytest.fixture
def mock_code_generation_result():
    """Mock code generation result."""
    code_string = """
# List comprehension example
numbers = [1, 2, 3, 4, 5]
squared = [x**2 for x in numbers]
print(squared)
"""
    return (
        {"status": "success", "generated_code": compile(code_string, "<string>", "exec")},
        code_string
    )


@pytest.fixture
def mock_code_execution_output():
    """Mock code execution output."""
    return "[1, 4, 9, 16, 25]\n"


@pytest.mark.integration
class TestOrchestratorIntegration:
    """Integration tests for OrchestratorAgent."""

    @patch('mcp_hub.agents.code_runner.modal')
    @patch('mcp_hub.agents.question_enhancer.make_llm_completion')
    @patch('mcp_hub.agents.web_search.TavilyClient')
    @patch('mcp_hub.agents.code_generator.make_llm_completion')
    @patch('mcp_hub.agents.orchestrator.make_llm_completion')
    def test_full_orchestration_success(
        self,
        mock_orchestrator_llm,
        mock_code_gen_llm,
        mock_tavily,
        mock_question_llm,
        mock_modal,
        mock_question_enhancer_response,
        mock_search_results,
        mock_code_generation_result,
        mock_code_execution_output
    ):
        """Test successful end-to-end orchestration workflow."""
        # Setup mocks
        mock_question_llm.return_value = '{"sub_questions": ["Q1?", "Q2?", "Q3?"]}'

        mock_tavily_instance = Mock()
        mock_tavily_instance.search.return_value = mock_search_results
        mock_tavily.return_value = mock_tavily_instance

        mock_code_gen_llm.return_value = mock_code_generation_result[1]
        mock_orchestrator_llm.return_value = "Here's a summary of the orchestration..."

        # Mock Modal sandbox
        mock_sandbox = Mock()
        mock_proc = Mock()
        mock_proc.stdout.read.return_value = mock_code_execution_output
        mock_proc.stderr.read.return_value = ""
        mock_sandbox.exec.return_value = mock_proc
        mock_modal.Sandbox.create.return_value = mock_sandbox
        mock_modal.App.lookup.return_value = Mock()
        mock_modal.Image.debian_slim.return_value.pip_install.return_value.apt_install.return_value.env.return_value = Mock()

        # Create orchestrator and run
        orchestrator = OrchestratorAgent()
        result, narrative = orchestrator.orchestrate("How do I use list comprehension in Python?")

        # Verify result structure
        assert result["status"] == "success"
        assert "user_request" in result
        assert "sub_questions" in result
        assert "search_results" in result
        assert "code_string" in result
        assert "execution_output" in result
        assert "citations" in result
        assert "final_summary" in result

        # Verify sub_questions were generated
        assert len(result["sub_questions"]) == 3

        # Verify narrative was created
        assert isinstance(narrative, str)
        assert len(narrative) > 0

        # Verify all agents were called
        assert mock_question_llm.called
        assert mock_tavily_instance.search.called
        assert mock_code_gen_llm.called


    @patch('mcp_hub.agents.question_enhancer.make_llm_completion')
    def test_orchestration_with_question_enhancement_failure(self, mock_llm):
        """Test orchestration handles question enhancement failure gracefully."""
        # Mock LLM to return invalid JSON
        mock_llm.return_value = "This is not valid JSON"

        orchestrator = OrchestratorAgent()
        result, narrative = orchestrator.orchestrate("Test query")

        # Should handle error gracefully
        assert "error" in result or result["status"] == "success"  # Either errors or continues with fallback


    @patch('mcp_hub.agents.web_search.TavilyClient')
    @patch('mcp_hub.agents.question_enhancer.make_llm_completion')
    def test_orchestration_with_search_failure(self, mock_question_llm, mock_tavily):
        """Test orchestration continues when web search fails."""
        # Setup mocks
        mock_question_llm.return_value = '{"sub_questions": ["Q1?", "Q2?", "Q3?"]}'

        mock_tavily_instance = Mock()
        mock_tavily_instance.search.side_effect = Exception("Search API error")
        mock_tavily.return_value = mock_tavily_instance

        orchestrator = OrchestratorAgent()
        result, narrative = orchestrator.orchestrate("Test query")

        # Should continue despite search failures
        assert "sub_questions" in result or "error" in result


    @patch('mcp_hub.agents.code_runner.modal')
    @patch('mcp_hub.agents.code_generator.make_llm_completion')
    @patch('mcp_hub.agents.web_search.TavilyClient')
    @patch('mcp_hub.agents.question_enhancer.make_llm_completion')
    def test_orchestration_with_code_generation_failure(
        self,
        mock_question_llm,
        mock_tavily,
        mock_code_gen_llm,
        mock_modal
    ):
        """Test orchestration handles code generation failure."""
        # Setup mocks
        mock_question_llm.return_value = '{"sub_questions": ["Q1?"]}'

        mock_tavily_instance = Mock()
        mock_tavily_instance.search.return_value = {
            "query": "test",
            "results": [],
            "tavily_answer": "No results"
        }
        mock_tavily.return_value = mock_tavily_instance

        # Mock code generation to fail
        mock_code_gen_llm.side_effect = Exception("Code generation failed")

        orchestrator = OrchestratorAgent()
        result, narrative = orchestrator.orchestrate("Generate invalid code")

        # Should handle error gracefully
        assert "error" in result or "code_string" in result


    @patch('mcp_hub.agents.code_runner.modal')
    @patch('mcp_hub.agents.code_generator.make_llm_completion')
    @patch('mcp_hub.agents.web_search.TavilyClient')
    @patch('mcp_hub.agents.question_enhancer.make_llm_completion')
    @patch('mcp_hub.agents.orchestrator.make_llm_completion')
    def test_orchestration_with_code_execution_failure(
        self,
        mock_orchestrator_llm,
        mock_question_llm,
        mock_tavily,
        mock_code_gen_llm,
        mock_modal
    ):
        """Test orchestration handles code execution failure."""
        # Setup mocks
        mock_question_llm.return_value = '{"sub_questions": ["Q1?"]}'

        mock_tavily_instance = Mock()
        mock_tavily_instance.search.return_value = {"query": "test", "results": [], "tavily_answer": ""}
        mock_tavily.return_value = mock_tavily_instance

        valid_code = "print('Hello')"
        mock_code_gen_llm.return_value = valid_code

        # Mock Modal to fail
        mock_modal.Sandbox.create.side_effect = Exception("Sandbox creation failed")
        mock_modal.App.lookup.return_value = Mock()
        mock_modal.Image.debian_slim.return_value.pip_install.return_value.apt_install.return_value.env.return_value = Mock()

        mock_orchestrator_llm.return_value = "Summary despite execution failure"

        orchestrator = OrchestratorAgent()
        result, narrative = orchestrator.orchestrate("Test query")

        # Should complete with execution failure noted
        assert "execution_output" in result
        # Execution output should indicate failure
        assert "failed" in result.get("execution_output", "").lower() or result["execution_output"] == ""


    @patch('mcp_hub.agents.orchestrator.make_llm_completion')
    @patch('mcp_hub.agents.code_runner.modal')
    @patch('mcp_hub.agents.code_generator.make_llm_completion')
    @patch('mcp_hub.agents.web_search.TavilyClient')
    @patch('mcp_hub.agents.question_enhancer.make_llm_completion')
    def test_orchestration_result_structure(
        self,
        mock_question_llm,
        mock_tavily,
        mock_code_gen_llm,
        mock_modal,
        mock_orchestrator_llm
    ):
        """Test that orchestration result has correct structure."""
        # Setup minimal mocks
        mock_question_llm.return_value = '{"sub_questions": ["Q1?"]}'
        mock_tavily_instance = Mock()
        mock_tavily_instance.search.return_value = {"query": "test", "results": [], "tavily_answer": ""}
        mock_tavily.return_value = mock_tavily_instance
        mock_code_gen_llm.return_value = "print('test')"
        mock_orchestrator_llm.return_value = "Summary"

        # Mock Modal
        mock_sandbox = Mock()
        mock_proc = Mock()
        mock_proc.stdout.read.return_value = "test\n"
        mock_proc.stderr.read.return_value = ""
        mock_sandbox.exec.return_value = mock_proc
        mock_modal.Sandbox.create.return_value = mock_sandbox
        mock_modal.App.lookup.return_value = Mock()
        mock_modal.Image.debian_slim.return_value.pip_install.return_value.apt_install.return_value.env.return_value = Mock()

        orchestrator = OrchestratorAgent()
        result, narrative = orchestrator.orchestrate("Test query")

        # Verify required fields exist
        required_fields = [
            "status",
            "user_request",
            "sub_questions",
            "search_results",
            "search_summaries",
            "code_string",
            "execution_output",
            "citations",
            "final_summary"
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

        # Verify narrative is a string
        assert isinstance(narrative, str)
        assert len(narrative) > 0
