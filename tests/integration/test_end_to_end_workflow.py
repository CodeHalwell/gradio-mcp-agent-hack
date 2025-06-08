"""Integration tests for MCP Hub end-to-end workflows."""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import sys
import os
import asyncio

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestEndToEndIntegration:
    """Test complete end-to-end workflows."""
    
    def test_full_sync_workflow_success(self):
        """Test complete synchronous workflow from request to execution."""
        with patch('modal.App') as mock_app_class, \
             patch('modal.Image'), \
             patch('tavily.TavilyClient') as mock_tavily_class, \
             patch('modal.Sandbox') as mock_sandbox_class:
            
            from app import OrchestratorAgent
            
            # Setup comprehensive mocks for full workflow
            mock_app = Mock()
            mock_app_class.lookup.return_value = mock_app
            
            mock_tavily = Mock()
            mock_tavily_class.return_value = mock_tavily
            mock_tavily.search.return_value = {
                "results": [
                    {
                        "title": "Python CSV Tutorial",
                        "url": "https://example.com/csv-tutorial",
                        "content": "Learn how to work with CSV files using pandas...",
                        "score": 0.95
                    },
                    {
                        "title": "Data Visualization Guide",
                        "url": "https://example.com/viz-guide", 
                        "content": "Create charts and graphs with matplotlib...",
                        "score": 0.87
                    }
                ],
                "answer": "Use pandas for CSV manipulation and matplotlib for visualization"
            }
            
            mock_sandbox = Mock()
            mock_sandbox_class.create.return_value = mock_sandbox
            mock_exec_result = Mock()
            mock_exec_result.stdout = "DataFrame loaded successfully\\nChart created and saved"
            mock_exec_result.stderr = ""
            mock_exec_result.returncode = 0
            mock_sandbox.exec.return_value = mock_exec_result
            
            # Execute full workflow
            orchestrator = OrchestratorAgent()
            result, summary = orchestrator.orchestrate(
                "Create a Python script to analyze CSV data and generate visualizations"
            )
            
            # Verify end-to-end success
            assert result["status"] == "success"
            assert "research_summary" in result
            assert "generated_code" in result
            assert "execution_result" in result
            assert "citations" in result
            
            # Verify workflow components
            assert "CSV" in result["research_summary"] or "csv" in result["research_summary"]
            assert "pandas" in result["generated_code"] or "CSV" in result["generated_code"]
            assert "DataFrame loaded successfully" in result["execution_result"]
            
            # Verify citations were generated
            assert len(result["citations"]) > 0
    
    @pytest.mark.asyncio
    async def test_full_async_workflow_success(self):
        """Test complete asynchronous workflow."""
        with patch('modal.App') as mock_app_class, \
             patch('modal.Image'), \
             patch('tavily.TavilyClient') as mock_tavily_class, \
             patch('app.WarmSandboxPool') as mock_pool_class:
            
            from app import OrchestratorAgent
            
            # Setup mocks
            mock_app = Mock()
            mock_app_class.lookup.return_value = mock_app
            
            mock_tavily = Mock()
            mock_tavily_class.return_value = mock_tavily
            mock_tavily.search.return_value = {
                "results": [{"title": "API Tutorial", "url": "https://example.com/api"}],
                "answer": "Use requests library for API calls"
            }
            
            mock_pool = AsyncMock()
            mock_pool.execute_code.return_value = "API call successful: 200 OK"
            mock_pool_class.return_value = mock_pool
            
            # Execute async workflow
            orchestrator = OrchestratorAgent()
            result, summary = await orchestrator.orchestrate_async(
                "Create a script to fetch data from a REST API"
            )
            
            # Verify async execution
            assert result["status"] == "success"
            assert "API call successful" in result["execution_result"]
            mock_pool.execute_code.assert_called_once()
    
    def test_workflow_with_search_failures(self):
        """Test workflow resilience when search fails."""
        with patch('modal.App') as mock_app_class, \
             patch('modal.Image'), \
             patch('tavily.TavilyClient') as mock_tavily_class, \
             patch('modal.Sandbox') as mock_sandbox_class:
            
            from app import OrchestratorAgent
            
            # Setup mocks with search failure
            mock_app = Mock()
            mock_app_class.lookup.return_value = mock_app
            
            mock_tavily = Mock()
            mock_tavily_class.return_value = mock_tavily
            mock_tavily.search.side_effect = Exception("Search service unavailable")
            
            mock_sandbox = Mock()
            mock_sandbox_class.create.return_value = mock_sandbox
            mock_exec_result = Mock()
            mock_exec_result.stdout = "Basic script executed"
            mock_exec_result.stderr = ""
            mock_exec_result.returncode = 0
            mock_sandbox.exec.return_value = mock_exec_result
            
            # Execute workflow
            orchestrator = OrchestratorAgent()
            result, summary = orchestrator.orchestrate("Create a basic Python script")
            
            # Should handle search failure gracefully
            # Implementation should fall back to generating code without web research
            assert "status" in result
    
    def test_workflow_with_code_execution_failure(self):
        """Test workflow when code execution fails."""
        with patch('modal.App') as mock_app_class, \
             patch('modal.Image'), \
             patch('tavily.TavilyClient') as mock_tavily_class, \
             patch('modal.Sandbox') as mock_sandbox_class:
            
            from app import OrchestratorAgent
            
            # Setup mocks
            mock_app = Mock()
            mock_app_class.lookup.return_value = mock_app
            
            mock_tavily = Mock()
            mock_tavily_class.return_value = mock_tavily
            mock_tavily.search.return_value = {
                "results": [{"title": "Tutorial", "url": "https://example.com"}],
                "answer": "Basic programming tutorial"
            }
            
            # Setup code execution failure
            mock_sandbox = Mock()
            mock_sandbox_class.create.return_value = mock_sandbox
            mock_exec_result = Mock()
            mock_exec_result.stdout = ""
            mock_exec_result.stderr = "SyntaxError: invalid syntax"
            mock_exec_result.returncode = 1
            mock_sandbox.exec.return_value = mock_exec_result
            
            # Execute workflow
            orchestrator = OrchestratorAgent()
            result, summary = orchestrator.orchestrate("Create a Python script")
            
            # Should handle execution failure
            assert "status" in result
            if result["status"] == "success":
                assert "SyntaxError" in result["execution_result"]
    
    def test_workflow_with_partial_component_failures(self):
        """Test workflow resilience with multiple component failures."""
        with patch('modal.App') as mock_app_class, \
             patch('modal.Image'), \
             patch('tavily.TavilyClient') as mock_tavily_class:
            
            from app import OrchestratorAgent
            
            # Setup mocks with various failures
            mock_app = Mock()
            mock_app_class.lookup.return_value = mock_app
            
            mock_tavily = Mock()
            mock_tavily_class.return_value = mock_tavily
            
            orchestrator = OrchestratorAgent()
            
            # Mock question enhancer failure
            orchestrator.question_enhancer.enhance_question = Mock(return_value={
                "error": "Enhancement failed",
                "sub_questions": []
            })
            
            # Mock search success
            orchestrator.web_search.search = Mock(return_value={
                "status": "success",
                "results": [],
                "tavily_answer": "Limited results"
            })
            
            # Mock LLM processor failure
            orchestrator.llm_processor.process = Mock(return_value={
                "status": "error",
                "error": "LLM processing failed"
            })
            
            # Mock code generator success
            orchestrator.code_generator.generate_code = Mock(return_value=[{
                "status": "success",
                "code": "print('Hello, World!')",
                "explanation": "Basic script"
            }, "Generated"])
            
            # Mock code runner failure
            orchestrator.code_runner.run_code = Mock(
                side_effect=Exception("Execution service unavailable")
            )
            
            # Mock citation formatter success
            orchestrator.citation_formatter.format_citations = Mock(return_value={
                "status": "success",
                "citations": []
            })
            
            # Execute workflow
            result, summary = orchestrator.orchestrate("Test request with multiple failures")
            
            # Should handle multiple failures gracefully
            assert "status" in result
            # Should still have some successful components
    
    def test_async_vs_sync_behavior_differences(self):
        """Test differences between async and sync orchestration."""
        with patch('modal.App') as mock_app_class, \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import OrchestratorAgent
            
            mock_app = Mock()
            mock_app_class.lookup.return_value = mock_app
            
            orchestrator = OrchestratorAgent()
            
            # Mock components for both sync and async
            orchestrator.question_enhancer.enhance_question = Mock(return_value={
                "sub_questions": ["Test question?"]
            })
            
            orchestrator.web_search.search = Mock(return_value={
                "status": "success",
                "results": [{"title": "Test", "url": "https://example.com"}],
                "tavily_answer": "Test answer"
            })
            
            orchestrator.llm_processor.process = Mock(return_value={
                "status": "success",
                "llm_processed_output": "Processed content"
            })
            
            orchestrator.code_generator.generate_code = Mock(return_value=[{
                "status": "success",
                "code": "print('test')",
                "explanation": "Test code"
            }, "Generated"])
            
            orchestrator.code_runner.run_code = Mock(return_value="Sync execution")
            orchestrator.code_runner.run_code_async = AsyncMock(return_value="Async execution")
            
            orchestrator.citation_formatter.format_citations = Mock(return_value={
                "status": "success",
                "citations": ["Test citation"]
            })
            
            # Test sync workflow
            sync_result, sync_summary = orchestrator.orchestrate("Test request")
            
            # Test async workflow
            async def test_async():
                return await orchestrator.orchestrate_async("Test request")
            
            async_result, async_summary = asyncio.run(test_async())
            
            # Compare behaviors
            assert sync_result["status"] == "success"
            assert async_result["status"] == "success"
            
            # Verify different execution methods were used
            assert "Sync execution" in sync_result["execution_result"]
            assert "Async execution" in async_result["execution_result"]
    
    def test_large_scale_workflow_performance(self):
        """Test workflow with large-scale data and complex requirements."""
        with patch('modal.App') as mock_app_class, \
             patch('modal.Image'), \
             patch('tavily.TavilyClient') as mock_tavily_class, \
             patch('modal.Sandbox') as mock_sandbox_class:
            
            from app import OrchestratorAgent
            
            # Setup mocks for large-scale scenario
            mock_app = Mock()
            mock_app_class.lookup.return_value = mock_app
            
            mock_tavily = Mock()
            mock_tavily_class.return_value = mock_tavily
            
            # Large search results
            large_results = [
                {
                    "title": f"Tutorial {i}",
                    "url": f"https://example.com/tutorial{i}",
                    "content": f"Tutorial content {i} " * 100,  # Large content
                    "score": 0.9 - (i * 0.1)
                }
                for i in range(10)
            ]
            
            mock_tavily.search.return_value = {
                "results": large_results,
                "answer": "Comprehensive answer " * 50  # Large answer
            }
            
            mock_sandbox = Mock()
            mock_sandbox_class.create.return_value = mock_sandbox
            mock_exec_result = Mock()
            mock_exec_result.stdout = "Large dataset processed successfully\\n" * 10
            mock_exec_result.stderr = ""
            mock_exec_result.returncode = 0
            mock_sandbox.exec.return_value = mock_exec_result
            
            # Execute large-scale workflow
            orchestrator = OrchestratorAgent()
            
            complex_request = """
            Create a comprehensive data analysis pipeline that:
            1. Reads multiple CSV files from different sources
            2. Performs data cleaning and validation
            3. Conducts statistical analysis
            4. Generates multiple types of visualizations
            5. Exports results in various formats
            6. Includes error handling and logging
            """
            
            result, summary = orchestrator.orchestrate(complex_request)
            
            # Verify handling of large-scale workflow
            assert "status" in result
            if result["status"] == "success":
                assert len(result["research_summary"]) > 100  # Substantial content
                assert len(result["generated_code"]) > 50  # Substantial code
    
    def test_workflow_error_recovery_and_fallbacks(self):
        """Test error recovery mechanisms and fallback behaviors."""
        with patch('modal.App') as mock_app_class, \
             patch('modal.Image'), \
             patch('tavily.TavilyClient') as mock_tavily_class:
            
            from app import OrchestratorAgent
            
            mock_app = Mock()
            mock_app_class.lookup.return_value = mock_app
            
            orchestrator = OrchestratorAgent()
            
            # Simulate systematic failures with recovery
            call_count = {"count": 0}
            
            def failing_then_succeeding(*args, **kwargs):
                call_count["count"] += 1
                if call_count["count"] <= 2:
                    raise Exception(f"Failure {call_count['count']}")
                return {
                    "status": "success",
                    "results": [{"title": "Recovery success", "url": "https://example.com"}],
                    "tavily_answer": "Recovered successfully"
                }
            
            orchestrator.web_search.search = Mock(side_effect=failing_then_succeeding)
            
            # Mock other successful components
            orchestrator.question_enhancer.enhance_question = Mock(return_value={
                "sub_questions": ["Recovery test?"]
            })
            
            orchestrator.llm_processor.process = Mock(return_value={
                "status": "success",
                "llm_processed_output": "Processed after recovery"
            })
            
            orchestrator.code_generator.generate_code = Mock(return_value=[{
                "status": "success",
                "code": "# Recovery code",
                "explanation": "Code generated after recovery"
            }, "Generated"])
            
            orchestrator.code_runner.run_code = Mock(return_value="Executed after recovery")
            
            orchestrator.citation_formatter.format_citations = Mock(return_value={
                "status": "success",
                "citations": ["Recovery citation"]
            })
            
            # Execute workflow with recovery
            result, summary = orchestrator.orchestrate("Test error recovery")
            
            # Verify recovery worked
            assert result["status"] == "success"
            assert "Recovered successfully" in result["research_summary"]
            assert "Executed after recovery" in result["execution_result"]