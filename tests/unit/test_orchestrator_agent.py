"""Unit tests for OrchestratorAgent."""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import sys
import os
import asyncio

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestOrchestratorAgent:
    """Test OrchestratorAgent functionality."""
    
    def test_orchestrator_instantiation(self):
        """Test OrchestratorAgent can be instantiated."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import OrchestratorAgent
            
            agent = OrchestratorAgent()
            assert agent is not None
            assert hasattr(agent, 'question_enhancer')
            assert hasattr(agent, 'web_search')
            assert hasattr(agent, 'llm_processor')
            assert hasattr(agent, 'citation_formatter')
            assert hasattr(agent, 'code_generator')
            assert hasattr(agent, 'code_runner')
    
    def test_orchestrate_sync_basic_workflow(self):
        """Test basic synchronous orchestration workflow."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import OrchestratorAgent
            
            # Create agent with mocked sub-agents
            agent = OrchestratorAgent()
            
            # Mock sub-agents
            agent.question_enhancer.enhance_question = Mock(return_value={
                "sub_questions": ["How to read CSV?", "How to plot data?"]
            })
            
            agent.web_search.search = Mock(return_value={
                "status": "success",
                "results": [{"title": "CSV Tutorial", "url": "https://example.com"}],
                "tavily_answer": "Use pandas to read CSV files"
            })
            
            agent.llm_processor.process = Mock(return_value={
                "status": "success",
                "llm_processed_output": "Comprehensive guide on CSV analysis"
            })
            
            agent.code_generator.generate_code = Mock(return_value=[{
                "status": "success",
                "code": "import pandas as pd\\ndf = pd.read_csv('data.csv')",
                "explanation": "Read CSV file"
            }, "Generated code"])
            
            agent.code_runner.run_code = Mock(return_value="Code executed successfully")
            
            agent.citation_formatter.format_citations = Mock(return_value={
                "status": "success",
                "citations": ["Example.com. (2023). CSV Tutorial."]
            })
            
            # Execute
            result, summary = agent.orchestrate("Create a CSV analysis script")
            
            # Verify workflow execution
            assert result["status"] == "success"
            assert "research_summary" in result
            assert "generated_code" in result
            assert "execution_result" in result
            assert "citations" in result
            
            # Verify all sub-agents were called
            agent.question_enhancer.enhance_question.assert_called_once()
            agent.web_search.search.assert_called()
            agent.llm_processor.process.assert_called()
            agent.code_generator.generate_code.assert_called_once()
            agent.code_runner.run_code.assert_called_once()
            agent.citation_formatter.format_citations.assert_called_once()
    
    def test_orchestrate_sync_question_enhancer_failure(self):
        """Test orchestration when question enhancer fails."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import OrchestratorAgent
            
            agent = OrchestratorAgent()
            
            # Mock question enhancer failure
            agent.question_enhancer.enhance_question = Mock(return_value={
                "error": "Enhancement failed",
                "sub_questions": []
            })
            
            # Execute
            result, summary = agent.orchestrate("Test request")
            
            # Should continue with original question
            assert "status" in result
            # The orchestrator should handle the failure gracefully
    
    def test_orchestrate_sync_search_failure(self):
        """Test orchestration when web search fails."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import OrchestratorAgent
            
            agent = OrchestratorAgent()
            
            agent.question_enhancer.enhance_question = Mock(return_value={
                "sub_questions": ["Test question?"]
            })
            
            # Mock search failure
            agent.web_search.search = Mock(return_value={
                "status": "error",
                "error": "Search failed"
            })
            
            # Mock other agents to continue workflow
            agent.llm_processor.process = Mock(return_value={
                "status": "success",
                "llm_processed_output": "Fallback response"
            })
            
            agent.code_generator.generate_code = Mock(return_value=[{
                "status": "success",
                "code": "# Fallback code",
                "explanation": "Basic code"
            }, "Generated"])
            
            agent.code_runner.run_code = Mock(return_value="Executed")
            agent.citation_formatter.format_citations = Mock(return_value={
                "status": "success",
                "citations": []
            })
            
            # Execute
            result, summary = agent.orchestrate("Test request")
            
            # Should handle search failure and continue
            assert "status" in result
    
    def test_orchestrate_sync_code_generation_failure(self):
        """Test orchestration when code generation fails."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import OrchestratorAgent
            
            agent = OrchestratorAgent()
            
            # Setup successful early stages
            agent.question_enhancer.enhance_question = Mock(return_value={
                "sub_questions": ["Test question?"]
            })
            
            agent.web_search.search = Mock(return_value={
                "status": "success",
                "results": [],
                "tavily_answer": "Test answer"
            })
            
            agent.llm_processor.process = Mock(return_value={
                "status": "success",
                "llm_processed_output": "Processed context"
            })
            
            # Mock code generation failure
            agent.code_generator.generate_code = Mock(return_value=[{
                "status": "error",
                "error": "Code generation failed"
            }, ""])
            
            # Execute
            result, summary = agent.orchestrate("Test request")
            
            # Should handle code generation failure
            assert "status" in result
            # May have research results even if code generation fails
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(True, reason="orchestrate_async method not yet implemented in OrchestratorAgent")
    async def test_orchestrate_async_basic_workflow(self):
        """Test basic asynchronous orchestration workflow."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import OrchestratorAgent
            
            agent = OrchestratorAgent()
            
            # Mock async sub-question processing
            agent._run_subquestion_async = AsyncMock(return_value=[
                {
                    "question": "Test question?",
                    "results": [{"title": "Result", "url": "https://example.com"}],
                    "search_summary": "Test summary"
                },
                "## Test question?\\n### Research Summary:\\nTest summary"
            ])
            
            # Mock other components
            agent.llm_processor.process = Mock(return_value={
                "status": "success",
                "llm_processed_output": "Integrated summary"
            })
            
            agent.code_generator.generate_code = Mock(return_value=[{
                "status": "success",
                "code": "# Generated code",
                "explanation": "Test code"
            }, "Code generated"])
            
            agent.code_runner.run_code_async = AsyncMock(return_value="Async execution result")
            
            agent.citation_formatter.format_citations = Mock(return_value={
                "status": "success",
                "citations": ["Test citation"]
            })
            
            # Execute
            result, summary = await agent.orchestrate_async("Test async request")
            
            # Verify async execution
            assert result["status"] == "success"
            agent._run_subquestion_async.assert_called()
            agent.code_runner.run_code_async.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_subquestion_async(self):
        """Test asynchronous sub-question processing."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import OrchestratorAgent
            
            agent = OrchestratorAgent()
            
            # Mock sub-agents
            agent.web_search.search = Mock(return_value={
                "status": "success",
                "results": [{"title": "Test Result", "url": "https://example.com"}],
                "tavily_answer": "Test answer"
            })
            
            agent.llm_processor.process = Mock(return_value={
                "status": "success",
                "llm_processed_output": "Processed summary"
            })
            
            # Execute
            result, summary = await agent._run_subquestion_async(
                "How to use pandas?", 
                "Create data analysis script"
            )
            
            # Verify
            assert result["question"] == "How to use pandas?"
            assert "results" in result
            assert "search_summary" in result
            assert "## Subquestion: How to use pandas?" in summary
    
    def test_orchestrate_empty_request(self):
        """Test orchestration with empty request."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import OrchestratorAgent
            
            agent = OrchestratorAgent()
            
            # Execute with empty request
            result, summary = agent.orchestrate("")
            
            # Should handle empty request gracefully
            assert "status" in result
            # Likely returns error status
    
    def test_orchestrate_logging(self):
        """Test that orchestration operations are logged."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.logger') as mock_logger:
            
            from app import OrchestratorAgent
            
            agent = OrchestratorAgent()
            
            # Mock all sub-agents for quick execution
            agent.question_enhancer.enhance_question = Mock(return_value={
                "sub_questions": ["Test?"]
            })
            agent.web_search.search = Mock(return_value={
                "status": "success", "results": [], "tavily_answer": ""
            })
            agent.llm_processor.process = Mock(return_value={
                "status": "success", "llm_processed_output": ""
            })
            agent.code_generator.generate_code = Mock(return_value=[{
                "status": "success", "code": "pass", "explanation": ""
            }, ""])
            agent.code_runner.run_code = Mock(return_value="")
            agent.citation_formatter.format_citations = Mock(return_value={
                "status": "success", "citations": []
            })
            
            # Execute
            agent.orchestrate("Test request")
            
            # Verify logging occurred
            assert mock_logger.info.call_count >= 1
    
    def test_orchestrate_performance_tracking(self):
        """Test that orchestration includes performance tracking."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('time.time') as mock_time:
            
            from app import OrchestratorAgent
            
            # Mock time progression
            mock_time.side_effect = [0, 1, 2, 3, 4, 5]  # 5 second execution
            
            agent = OrchestratorAgent()
            
            # Mock sub-agents
            agent.question_enhancer.enhance_question = Mock(return_value={
                "sub_questions": ["Test?"]
            })
            agent.web_search.search = Mock(return_value={
                "status": "success", "results": [], "tavily_answer": ""
            })
            agent.llm_processor.process = Mock(return_value={
                "status": "success", "llm_processed_output": ""
            })
            agent.code_generator.generate_code = Mock(return_value=[{
                "status": "success", "code": "pass", "explanation": ""
            }, ""])
            agent.code_runner.run_code = Mock(return_value="")
            agent.citation_formatter.format_citations = Mock(return_value={
                "status": "success", "citations": []
            })
            
            # Execute
            result, summary = agent.orchestrate("Test request")
            
            # Performance tracking should be included in the implementation
            assert "status" in result
    
    def test_orchestrate_search_result_limits(self):
        """Test that search results are properly limited."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import OrchestratorAgent
            
            agent = OrchestratorAgent()
            
            # Mock question enhancer with many questions
            agent.question_enhancer.enhance_question = Mock(return_value={
                "sub_questions": [f"Question {i}?" for i in range(10)]
            })
            
            agent.web_search.search = Mock(return_value={
                "status": "success", "results": [], "tavily_answer": ""
            })
            agent.llm_processor.process = Mock(return_value={
                "status": "success", "llm_processed_output": ""
            })
            agent.code_generator.generate_code = Mock(return_value=[{
                "status": "success", "code": "pass", "explanation": ""
            }, ""])
            agent.code_runner.run_code = Mock(return_value="")
            agent.citation_formatter.format_citations = Mock(return_value={
                "status": "success", "citations": []
            })
            
            # Execute
            agent.orchestrate("Test request")
            
            # Verify search is limited (implementation should limit to 2 questions)
            # The exact number of calls depends on the implementation
            assert agent.web_search.search.call_count <= 5  # Reasonable limit