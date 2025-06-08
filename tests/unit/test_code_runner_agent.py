"""Unit tests for CodeRunnerAgent."""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import sys
import os
import asyncio

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestCodeRunnerAgent:
    """Test CodeRunnerAgent functionality."""
    
    def test_code_runner_instantiation(self):
        """Test CodeRunnerAgent can be instantiated."""
        with patch('modal.App') as mock_app_class, \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import CodeRunnerAgent
            
            # Setup Mock app
            mock_app = Mock()
            mock_app_class.lookup.return_value = mock_app
            
            agent = CodeRunnerAgent()
            assert agent is not None
            assert agent.app == mock_app
    
    def test_analyze_code_dependencies(self):
        """Test code dependency analysis."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('mcp_hub.package_utils.extract_imports_from_code') as mock_extract, \
             patch('mcp_hub.package_utils.get_packages_to_install') as mock_packages:
            
            from app import CodeRunnerAgent
            
            # Setup mocks
            mock_extract.return_value = ["pandas", "numpy", "matplotlib"]
            mock_packages.return_value = ["pandas", "matplotlib"]  # numpy is built-in
            
            # Execute
            agent = CodeRunnerAgent()
            code = "import pandas as pd\\nimport numpy as np\\nimport matplotlib.pyplot as plt"
            dependencies = agent._analyze_code_dependencies(code)
            
            # Verify
            assert dependencies == ["pandas", "matplotlib"]
            mock_extract.assert_called_once_with(code)
            mock_packages.assert_called_once()
    
    def test_analyze_code_dependencies_error_handling(self):
        """Test error handling in dependency analysis."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('mcp_hub.package_utils.extract_imports_from_code') as mock_extract:
            
            from app import CodeRunnerAgent
            
            # Setup error
            mock_extract.side_effect = Exception("Analysis failed")
            
            # Execute
            agent = CodeRunnerAgent()
            dependencies = agent._analyze_code_dependencies("test code")
            
            # Should return empty list on error
            assert dependencies == []
    
    def test_run_code_sync_basic(self):
        """Test basic synchronous code execution."""
        with patch('modal.App') as mock_app_class, \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('modal.Sandbox') as mock_sandbox_class:
            
            from app import CodeRunnerAgent
            
            # Setup mocks
            mock_app = Mock()
            mock_app_class.lookup.return_value = mock_app
            
            mock_sandbox = Mock()
            mock_sandbox_class.create.return_value = mock_sandbox
            
            # Setup execution result
            mock_exec_result = Mock()
            mock_exec_result.stdout = "Hello, World!"
            mock_exec_result.stderr = ""
            mock_exec_result.returncode = 0
            mock_sandbox.exec.return_value = mock_exec_result
            
            # Execute
            agent = CodeRunnerAgent()
            result = agent.run_code("print('Hello, World!')")
            
            # Verify
            assert "Hello, World!" in result
            mock_sandbox.exec.assert_called_once()
    
    def test_run_code_sync_with_error(self):
        """Test synchronous code execution with runtime error."""
        with patch('modal.App') as mock_app_class, \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('modal.Sandbox') as mock_sandbox_class:
            
            from app import CodeRunnerAgent
            
            # Setup mocks
            mock_app = Mock()
            mock_app_class.lookup.return_value = mock_app
            
            mock_sandbox = Mock()
            mock_sandbox_class.create.return_value = mock_sandbox
            
            # Setup execution error
            mock_exec_result = Mock()
            mock_exec_result.stdout = ""
            mock_exec_result.stderr = "NameError: name 'undefined_var' is not defined"
            mock_exec_result.returncode = 1
            mock_sandbox.exec.return_value = mock_exec_result
            
            # Execute
            agent = CodeRunnerAgent()
            result = agent.run_code("print(undefined_var)")
            
            # Verify error handling
            assert "NameError" in result
            assert "undefined_var" in result
    
    def test_run_code_sync_modal_error(self):
        """Test handling of Modal service errors."""
        with patch('modal.App') as mock_app_class, \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('modal.Sandbox') as mock_sandbox_class:
            
            from app import CodeRunnerAgent
            
            # Setup mocks
            mock_app = Mock()
            mock_app_class.lookup.return_value = mock_app
            
            # Setup Modal error
            mock_sandbox_class.create.side_effect = Exception("Modal service error")
            
            # Execute
            agent = CodeRunnerAgent()
            result = agent.run_code("print('test')")
            
            # Verify error handling
            assert "Modal service error" in result or "Failed to execute" in result
    
    @pytest.mark.asyncio
    async def test_run_code_async_basic(self):
        """Test basic asynchronous code execution."""
        with patch('modal.App') as mock_app_class, \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import CodeRunnerAgent
            
            # Setup mocks
            mock_app = Mock()
            mock_app_class.lookup.return_value = mock_app
            
            # Create agent
            agent = CodeRunnerAgent()
            
            # Mock the pool to return successful execution
            agent.sandbox_pool = Mock()
            
            # Create an async context manager mock for the sandbox
            mock_sandbox = Mock()
            mock_sandbox.exec = Mock(return_value="Async execution result")
            
            # Create a proper async context manager
            class MockAsyncContextManager:
                async def __aenter__(self):
                    return mock_sandbox
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass
            
            # Make get_sandbox() return the async context manager directly
            agent.sandbox_pool.get_sandbox = Mock(return_value=MockAsyncContextManager())
            agent._pool_initialized = True
            
            # Execute
            result = await agent.run_code_async("print('Hello Async!')")
            
            # Verify - the result will be processed text, not the exact mock return
            assert result is not None
            # Check that the sandbox exec was called (which is how code gets executed)
            assert mock_sandbox.exec.call_count >= 1  # Called for bash and python commands
    
    @pytest.mark.asyncio
    async def test_run_code_async_pool_initialization(self):
        """Test sandbox pool initialization during async execution."""
        with patch('modal.App') as mock_app_class, \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('app.WarmSandboxPool') as mock_pool_class:
            
            from app import CodeRunnerAgent
            
            # Setup mocks
            mock_app = Mock()
            mock_app_class.lookup.return_value = mock_app
            
            mock_pool = AsyncMock()
            mock_pool.execute_code.return_value = "Pool execution result"
            mock_pool_class.return_value = mock_pool
            
            # Execute
            agent = CodeRunnerAgent()
            result = await agent.run_code_async("print('test')")
            
            # Verify pool initialization
            mock_pool.start.assert_called_once()
            assert agent._pool_initialized
    
    def test_add_safety_shim(self):
        """Test security shim addition to code."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import CodeRunnerAgent
            
            agent = CodeRunnerAgent()
            original_code = "print('Hello, World!')"
            
            # Execute
            shim_code = agent._add_safety_shim(original_code)
            
            # Verify security features
            assert "RESTRICTED_BUILTINS" in shim_code
            assert "print('Hello, World!')" in shim_code
            assert "import" in shim_code  # Safety imports
    
    def test_indent_code(self):
        """Test code indentation utility."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import CodeRunnerAgent
            
            agent = CodeRunnerAgent()
            code = "line1\\nline2\\nline3"
            
            # Execute
            indented = agent._indent_code(code, indent=4)
            
            # Verify indentation
            lines = indented.split('\\n')
            for line in lines:
                if line.strip():  # Non-empty lines should be indented
                    assert line.startswith('    ')
    
    @pytest.mark.asyncio
    async def test_pool_stats(self):
        """Test getting sandbox pool statistics."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import CodeRunnerAgent
            
            agent = CodeRunnerAgent()
            
            # Test when pool is not initialized
            stats = await agent.get_pool_stats()
            assert "error" in stats
            
            # Test when pool is initialized
            mock_pool = Mock()
            mock_pool.get_stats.return_value = {
                "active_sandboxes": 3,
                "total_executions": 10
            }
            agent.sandbox_pool = mock_pool
            
            stats = await agent.get_pool_stats()
            assert "active_sandboxes" in stats
            assert stats["active_sandboxes"] == 3
    
    @pytest.mark.asyncio
    async def test_cleanup_pool(self):
        """Test sandbox pool cleanup."""
        with patch('modal.App'), \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'):
            
            from app import CodeRunnerAgent
            
            agent = CodeRunnerAgent()
            
            # Setup mock pool
            mock_pool = AsyncMock()
            agent.sandbox_pool = mock_pool
            
            # Execute cleanup
            await agent.cleanup_pool()
            
            # Verify cleanup
            mock_pool.shutdown.assert_called_once()
    
    def test_run_code_with_dependencies(self):
        """Test code execution with package dependencies."""
        with patch('modal.App') as mock_app_class, \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('modal.Sandbox') as mock_sandbox_class, \
             patch.object(CodeRunnerAgent, '_analyze_code_dependencies') as mock_analyze:
            
            from app import CodeRunnerAgent
            
            # Setup mocks
            mock_app = Mock()
            mock_app_class.lookup.return_value = mock_app
            
            mock_sandbox = Mock()
            mock_sandbox_class.create.return_value = mock_sandbox
            
            mock_exec_result = Mock()
            mock_exec_result.stdout = "Package installed and code executed"
            mock_exec_result.stderr = ""
            mock_exec_result.returncode = 0
            mock_sandbox.exec.return_value = mock_exec_result
            
            # Setup dependency analysis
            mock_analyze.return_value = ["pandas"]
            
            # Execute
            agent = CodeRunnerAgent()
            result = agent.run_code("import pandas as pd\\nprint(pd.__version__)")
            
            # Verify
            mock_analyze.assert_called_once()
            assert "Package installed and code executed" in result
    
    def test_run_code_performance_tracking(self):
        """Test that code execution is tracked for performance."""
        with patch('modal.App') as mock_app_class, \
             patch('modal.Image'), \
             patch('tavily.TavilyClient'), \
             patch('modal.Sandbox') as mock_sandbox_class, \
             patch('app.logger') as mock_logger:
            
            from app import CodeRunnerAgent
            
            # Setup mocks
            mock_app = Mock()
            mock_app_class.lookup.return_value = mock_app
            
            mock_sandbox = Mock()
            mock_sandbox_class.create.return_value = mock_sandbox
            
            mock_exec_result = Mock()
            mock_exec_result.stdout = "Execution completed"
            mock_exec_result.stderr = ""
            mock_exec_result.returncode = 0
            mock_sandbox.exec.return_value = mock_exec_result
            
            # Execute
            agent = CodeRunnerAgent()
            agent.run_code("print('performance test')")
            
            # Verify logging (performance tracking should log)
            assert mock_logger.info.call_count >= 1