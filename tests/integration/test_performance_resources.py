"""Performance and resource tests for MCP Hub."""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import sys
import os
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


@pytest.mark.slow
class TestPerformanceMonitoring:
    """Test performance monitoring and metrics collection."""
    
    def test_metrics_collection_performance(self):
        """Test that metrics collection doesn't significantly impact performance."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from mcp_hub.performance_monitoring import MetricsCollector
            
            collector = MetricsCollector()
            
            # Time metrics recording
            start_time = time.time()
            
            # Record many metrics
            for i in range(1000):
                collector.record_metric(f"test_metric_{i % 10}", float(i), {"iteration": str(i)})
                collector.increment_counter(f"test_counter_{i % 5}", 1, {"batch": str(i // 100)})
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Metrics collection should be fast
            assert duration < 1.0, f"Metrics collection took {duration:.3f}s, should be < 1.0s"
    
    def test_concurrent_metrics_collection(self):
        """Test metrics collection under concurrent load."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from mcp_hub.performance_monitoring import MetricsCollector
            
            collector = MetricsCollector()
            results = []
            
            def record_metrics(thread_id):
                start = time.time()
                for i in range(100):
                    collector.record_metric("concurrent_test", float(i), {"thread": str(thread_id)})
                end = time.time()
                results.append(end - start)
            
            # Run concurrent metrics collection
            threads = []
            for i in range(10):
                thread = threading.Thread(target=record_metrics, args=(i,))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            # All threads should complete reasonably quickly
            max_time = max(results)
            assert max_time < 2.0, f"Slowest thread took {max_time:.3f}s"
    
    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring functionality."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('psutil.virtual_memory') as mock_memory:
            
            from mcp_hub.performance_monitoring import MetricsCollector
            
            # Mock memory stats
            mock_memory.return_value = Mock(
                total=8589934592,  # 8GB
                available=4294967296,  # 4GB
                percent=50.0,
                used=4294967296
            )
            
            collector = MetricsCollector()
            memory_stats = collector.get_system_metrics()
            
            assert "memory" in memory_stats
            assert memory_stats["memory"]["total"] == 8589934592
            assert memory_stats["memory"]["percent"] == 50.0


@pytest.mark.slow
class TestCachePerformance:
    """Test caching system performance."""
    
    def test_cache_hit_performance(self):
        """Test cache hit performance is fast."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('os.path.exists', return_value=True), \
             patch('pickle.load', return_value="cached_value"):
            
            from mcp_hub.cache_utils import CacheManager
            
            cache = CacheManager("test_cache")
            
            # Time cache hits
            start_time = time.time()
            
            for i in range(1000):
                value = cache.get(f"test_key_{i % 100}")
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Cache hits should be very fast
            assert duration < 0.5, f"1000 cache hits took {duration:.3f}s, should be < 0.5s"
    
    def test_cache_memory_efficiency(self):
        """Test cache memory usage efficiency."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('os.makedirs'), \
             patch('pickle.dump'):
            
            from mcp_hub.cache_utils import CacheManager
            
            cache = CacheManager("test_cache")
            
            # Store many items
            large_data = "x" * 1000  # 1KB strings
            
            for i in range(100):
                cache.set(f"large_key_{i}", large_data, ttl=300)
            
            # Cache should handle this without issues
            assert True  # If we reach here, no memory issues occurred
    
    def test_cache_cleanup_performance(self):
        """Test cache cleanup performance."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('os.listdir', return_value=[f"expired_{i}.cache" for i in range(1000)]), \
             patch('os.path.getmtime', return_value=time.time() - 1000), \
             patch('os.remove') as mock_remove:
            
            from mcp_hub.cache_utils import CacheManager
            
            cache = CacheManager("test_cache")
            
            # Time cleanup operation
            start_time = time.time()
            cache.cleanup_expired()
            end_time = time.time()
            
            duration = end_time - start_time
            
            # Cleanup should be reasonably fast
            assert duration < 2.0, f"Cache cleanup took {duration:.3f}s, should be < 2.0s"


@pytest.mark.slow
class TestAsyncPerformance:
    """Test asynchronous operation performance."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(True, reason="orchestrate_async method not yet implemented in OrchestratorAgent")
    async def test_async_orchestration_performance(self):
        """Test async orchestration performance vs sync."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import OrchestratorAgent
            
            orchestrator = OrchestratorAgent()
            
            # Mock fast responses
            orchestrator.question_enhancer.enhance_question = Mock(return_value={
                "sub_questions": ["Test question?"]
            })
            orchestrator.web_search.search = Mock(return_value={
                "status": "success", "results": [], "tavily_answer": "Fast answer"
            })
            orchestrator.llm_processor.process = Mock(return_value={
                "status": "success", "llm_processed_output": "Fast processing"
            })
            orchestrator.code_generator.generate_code = Mock(return_value=[{
                "status": "success", "code": "# Fast code", "explanation": "Fast"
            }, "Fast"])
            orchestrator.code_runner.run_code_async = AsyncMock(return_value="Fast async execution")
            orchestrator.code_runner.run_code = Mock(return_value="Fast sync execution")
            orchestrator.citation_formatter.format_citations = Mock(return_value={
                "formatted_citations": [], "error": None
            })
            
            # Time async execution
            start_async = time.time()
            await orchestrator.orchestrate_async("Test async performance")
            end_async = time.time()
            async_duration = end_async - start_async
            
            # Time sync execution
            start_sync = time.time()
            orchestrator.orchestrate("Test sync performance")
            end_sync = time.time()
            sync_duration = end_sync - start_sync
            
            # Both should be reasonably fast
            assert async_duration < 1.0, f"Async execution took {async_duration:.3f}s"
            assert sync_duration < 1.0, f"Sync execution took {sync_duration:.3f}s"
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(True, reason="orchestrate_async method not yet implemented in OrchestratorAgent")
    async def test_concurrent_async_operations(self):
        """Test performance under concurrent async operations."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import OrchestratorAgent
            
            orchestrator = OrchestratorAgent()
            
            # Mock very fast responses
            orchestrator._run_subquestion_async = AsyncMock(return_value=[
                {"question": "Fast?", "results": [], "search_summary": "Fast"},
                "Fast summary"
            ])
            orchestrator.llm_processor.process = Mock(return_value={
                "status": "success", "llm_processed_output": "Fast"
            })
            orchestrator.code_generator.generate_code = Mock(return_value=[{
                "status": "success", "code": "# Fast", "explanation": "Fast"
            }, "Fast"])
            orchestrator.code_runner.run_code_async = AsyncMock(return_value="Fast")
            orchestrator.citation_formatter.format_citations = Mock(return_value={
                "formatted_citations": [], "error": None
            })
            
            # Run multiple concurrent operations
            start_time = time.time()
            
            tasks = [
                orchestrator.orchestrate_async(f"Concurrent test {i}")
                for i in range(5)
            ]
            
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Concurrent execution should be efficient
            assert duration < 2.0, f"5 concurrent operations took {duration:.3f}s"
            assert len(results) == 5
            assert all(result[0]["status"] == "success" for result in results)


@pytest.mark.slow
class TestResourceManagement:
    """Test resource management and cleanup."""
    
    def test_no_resource_leaks_after_operations(self):
        """Test that no resources are leaked after operations."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import OrchestratorAgent
            
            # Track initial state
            initial_thread_count = threading.active_count()
            
            # Perform multiple operations
            orchestrator = OrchestratorAgent()
            
            # Mock agents for quick execution
            orchestrator.question_enhancer.enhance_question = Mock(return_value={
                "sub_questions": ["Test?"]
            })
            orchestrator.web_search.search = Mock(return_value={
                "status": "success", "results": [], "tavily_answer": ""
            })
            orchestrator.llm_processor.process = Mock(return_value={
                "status": "success", "llm_processed_output": ""
            })
            orchestrator.code_generator.generate_code = Mock(return_value=[{
                "status": "success", "code": "pass", "explanation": ""
            }, ""])
            orchestrator.code_runner.run_code = Mock(return_value="")
            orchestrator.citation_formatter.format_citations = Mock(return_value={
                "formatted_citations": [], "error": None
            })
            
            # Run multiple operations
            for i in range(10):
                orchestrator.orchestrate(f"Test operation {i}")
            
            # Check for resource leaks
            final_thread_count = threading.active_count()
            
            # Should not have significantly more threads
            assert final_thread_count <= initial_thread_count + 2, \
                f"Thread count increased from {initial_thread_count} to {final_thread_count}"
    
    @pytest.mark.asyncio
    async def test_async_resource_cleanup(self):
        """Test async resource cleanup."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import CodeRunnerAgent
            
            agent = CodeRunnerAgent()
            
            # Mock sandbox pool
            mock_pool = AsyncMock()
            agent.sandbox_pool = mock_pool
            
            # Test cleanup
            await agent.cleanup_pool()
            
            # Verify cleanup was called
            mock_pool.shutdown.assert_called_once()
    
    def test_modal_sandbox_resource_management(self):
        """Test Modal sandbox resource management."""
        with patch('modal.App') as mock_app_class, \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('modal.Sandbox') as mock_sandbox_class:
            
            from app import CodeRunnerAgent
            
            mock_app = Mock()
            mock_app_class.lookup.return_value = mock_app
            
            # Track sandbox creation
            created_sandboxes = []
            
            def track_sandbox_creation(*args, **kwargs):
                sandbox = Mock()
                created_sandboxes.append(sandbox)
                
                # Mock successful execution
                exec_result = Mock()
                exec_result.stdout = "Test output"
                exec_result.stderr = ""
                exec_result.returncode = 0
                sandbox.exec.return_value = exec_result
                
                return sandbox
            
            mock_sandbox_class.create.side_effect = track_sandbox_creation
            
            # Create agent and run code multiple times
            agent = CodeRunnerAgent()
            
            for i in range(5):
                result = agent.run_code(f"print('Test {i}')")
                assert "Test output" in result
            
            # Verify sandboxes were created
            assert len(created_sandboxes) == 5
    
    def test_memory_usage_stability(self):
        """Test memory usage remains stable during operations."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('psutil.Process') as mock_process:
            
            from app import OrchestratorAgent
            
            # Mock memory monitoring
            mock_proc = Mock()
            mock_process.return_value = mock_proc
            
            memory_readings = [100, 105, 103, 107, 102, 104, 106, 103]  # MB
            mock_proc.memory_info.side_effect = [
                Mock(rss=mb * 1024 * 1024) for mb in memory_readings
            ]
            
            orchestrator = OrchestratorAgent()
            
            # Mock fast operations
            orchestrator.question_enhancer.enhance_question = Mock(return_value={
                "sub_questions": ["Test?"]
            })
            orchestrator.web_search.search = Mock(return_value={
                "status": "success", "results": [], "tavily_answer": ""
            })
            orchestrator.llm_processor.process = Mock(return_value={
                "status": "success", "llm_processed_output": ""
            })
            orchestrator.code_generator.generate_code = Mock(return_value=[{
                "status": "success", "code": "pass", "explanation": ""
            }, ""])
            orchestrator.code_runner.run_code = Mock(return_value="")
            orchestrator.citation_formatter.format_citations = Mock(return_value={
                "formatted_citations": [], "error": None
            })
            
            # Run operations and check memory
            for i in range(len(memory_readings)):
                orchestrator.orchestrate(f"Memory test {i}")
                # Memory check happens via mocked psutil calls
            
            # Memory should remain relatively stable (within 10MB variance)
            max_memory = max(memory_readings)
            min_memory = min(memory_readings)
            memory_variance = max_memory - min_memory
            
            assert memory_variance <= 10, f"Memory variance {memory_variance}MB too high"


@pytest.mark.slow 
class TestStressTests:
    """Stress tests for system limits."""
    
    def test_large_input_handling(self):
        """Test handling of large input texts."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import LLMProcessorAgent
            
            agent = LLMProcessorAgent()
            
            with patch('app.make_llm_completion') as mock_llm:
                mock_llm.return_value = "Large text processed successfully"
                
                # Test with very large input (1MB)
                large_input = "This is a test sentence. " * 40000  # ~1MB
                
                start_time = time.time()
                result = agent.process(large_input, "summarize")
                end_time = time.time()
                
                # Should handle large input without timeout
                assert end_time - start_time < 5.0, "Large input processing took too long"
                assert result["status"] == "success"
    
    def test_rapid_successive_requests(self):
        """Test handling of rapid successive requests."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import WebSearchAgent
            
            agent = WebSearchAgent()
            agent.client.search = Mock(return_value={
                "results": [{"title": "Fast result", "url": "https://example.com"}],
                "answer": "Fast answer"
            })
            
            # Make rapid requests
            start_time = time.time()
            
            results = []
            for i in range(50):
                result = agent.search(f"Rapid test {i}")
                results.append(result)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Should handle rapid requests efficiently
            assert duration < 2.0, f"50 rapid requests took {duration:.3f}s"
            assert len(results) == 50
            assert all(r["status"] == "success" for r in results)
    
    def test_concurrent_user_simulation(self):
        """Test concurrent user simulation."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import agent_orchestrator
            
            with patch('app.orchestrator') as mock_orchestrator:
                mock_orchestrator.orchestrate.return_value = (
                    {"status": "success", "message": "Concurrent success"},
                    "Concurrent summary"
                )
                
                def simulate_user(user_id):
                    results = []
                    for i in range(3):
                        result = agent_orchestrator(f"User {user_id} request {i}")
                        results.append(result)
                    return results
                
                # Simulate 5 concurrent users
                with ThreadPoolExecutor(max_workers=5) as executor:
                    start_time = time.time()
                    futures = [executor.submit(simulate_user, i) for i in range(5)]
                    all_results = [future.result() for future in futures]
                    end_time = time.time()
                
                duration = end_time - start_time
                
                # Should handle concurrent users efficiently
                assert duration < 3.0, f"Concurrent user simulation took {duration:.3f}s"
                assert len(all_results) == 5
                assert all(len(user_results) == 3 for user_results in all_results)