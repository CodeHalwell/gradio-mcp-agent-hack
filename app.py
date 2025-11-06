"""
Enhanced MCP Hub - Single Unified Version with Advanced Features.

This module provides a comprehensive MCP (Model Context Protocol) Hub that integrates
multiple AI agents for research, code generation, and execution. It includes web search,
question enhancement, LLM processing, code generation, and secure code execution capabilities.

The hub is designed to be used as both a Gradio web interface and as an MCP server,
providing a unified API for AI-assisted development workflows.
"""
import gradio as gr
import modal
import textwrap
import base64
import marshal
import types
import time
import asyncio
import aiohttp
import ast
import json
from typing import Dict, Any, List
from functools import wraps
from contextlib import asynccontextmanager

# Import our custom modules
from mcp_hub.config import api_config, model_config, app_config
from mcp_hub.exceptions import APIError, ValidationError, CodeGenerationError, CodeExecutionError
from mcp_hub.utils import (
    validate_non_empty_string, extract_json_from_text,
    extract_urls_from_text, make_llm_completion,
    create_apa_citation
)
from mcp_hub.logging_config import logger
from tavily import TavilyClient

# Import advanced features with graceful fallback
ADVANCED_FEATURES_AVAILABLE = False
try:
    from mcp_hub.performance_monitoring import metrics_collector, track_performance, track_api_call
    from mcp_hub.cache_utils import cached
    from mcp_hub.reliability_utils import rate_limited, circuit_protected
    from mcp_hub.health_monitoring import health_monitor
    ADVANCED_FEATURES_AVAILABLE = True
    logger.info("Advanced features loaded successfully")
    
except ImportError as e:
    logger.info(f"Advanced features not available: {e}")
    logger.info("Running with basic features only")
    
    # Create dummy decorators for backward compatibility
    def track_performance(operation_name: str = None):
        def decorator(func): 
            return func
        return decorator
    
    def track_api_call(service_name: str):
        def decorator(func): 
            return func
        return decorator
    
    def rate_limited(service: str = "default", timeout: float = 10.0):
        def decorator(func): 
            return func
        return decorator
    
    def circuit_protected(service: str = "default"):
        def decorator(func): 
            return func
        return decorator
    
    def cached(ttl: int = 300):
        def decorator(func): 
            return func
        return decorator

# Performance tracking wrapper
def with_performance_tracking(operation_name: str):
    """
    Add performance tracking and metrics collection to any function (sync or async).

    This decorator wraps both synchronous and asynchronous functions to collect
    execution time, success/failure metrics, and error counts. It integrates with
    the advanced monitoring system when available.

    Args:
        operation_name (str): The name of the operation to track in metrics

    Returns:
        function: A decorator function that can wrap sync or async functions
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
                    success = False
                    error = str(e)
                    raise
                finally:
                    duration = time.time() - start_time
                    if ADVANCED_FEATURES_AVAILABLE:
                        metrics_collector.record_metric(f"{operation_name}_duration", duration, 
                                                        {"success": str(success), "operation": operation_name})
                        if not success:
                            metrics_collector.increment_counter(f"{operation_name}_errors", 1, 
                                                              {"operation": operation_name, "error": error})
                    logger.info(f"Operation {operation_name} completed in {duration:.2f}s (success: {success})")
                return result
            return async_wrapper
        else:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    success = False
                    error = str(e)
                    raise
                finally:
                    duration = time.time() - start_time
                    if ADVANCED_FEATURES_AVAILABLE:
                        metrics_collector.record_metric(f"{operation_name}_duration", duration, 
                                                        {"success": str(success), "operation": operation_name})
                        if not success:
                            metrics_collector.increment_counter(f"{operation_name}_errors", 1, 
                                                              {"operation": operation_name, "error": error})
                    logger.info(f"Operation {operation_name} completed in {duration:.2f}s (success: {success})")
                return result
            return wrapper
    return decorator


# Import all agents from the new modular structure
from mcp_hub.agents import (
    QuestionEnhancerAgent,
    WebSearchAgent,
    LLMProcessorAgent,
    CitationFormatterAgent,
    CodeGeneratorAgent,
    CodeRunnerAgent,
    OrchestratorAgent
)

# Initialize individual agents
question_enhancer = QuestionEnhancerAgent()
web_search = WebSearchAgent()
llm_processor = LLMProcessorAgent()
citation_formatter = CitationFormatterAgent()
code_generator = CodeGeneratorAgent()
code_runner = CodeRunnerAgent()

# Initialize orchestrator
orchestrator = OrchestratorAgent()

# ----------------------------------------
# Advanced Feature Functions
# ----------------------------------------

# Wrapper functions for backward compatibility with existing Gradio interface
def agent_orchestrator(user_request: str) -> tuple:
    """
    Wrapper for OrchestratorAgent with async-first approach and sync fallback.

    Provides a unified interface to the orchestrator that attempts async execution
    for better performance and falls back to synchronous execution if needed.
    Handles event loop management and thread pooling automatically.

    Args:
        user_request (str): The user's request to be processed

    Returns:
        tuple: A tuple containing the orchestration result and summary
    """
    try:
        # Try async orchestration first for better performance
        if hasattr(orchestrator, "orchestrate_async"):
            try:
                # Check if we're in an async context
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is already running (like in Gradio), we need to handle this differently
                    # Use asyncio.run_coroutine_threadsafe or run in thread pool
                    import concurrent.futures
                    
                    def run_async_in_thread():
                        # Create a new event loop for this thread
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(orchestrator.orchestrate_async(user_request))
                        finally:
                            new_loop.close()
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_async_in_thread)
                        result = future.result()
                else:
                    # No loop running, safe to use run_until_complete
                    result = loop.run_until_complete(orchestrator.orchestrate_async(user_request))
                
                logger.info("Successfully used async orchestration")
                return result
                
            except RuntimeError as e:
                if "cannot be called from a running event loop" in str(e):
                    logger.warning("Cannot use asyncio.run from running event loop, trying thread approach")
                    # Fallback: run in a separate thread
                    import concurrent.futures
                    
                    def run_async_in_thread():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(orchestrator.orchestrate_async(user_request))
                        finally:
                            new_loop.close()
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_async_in_thread)
                        return future.result()
                else:
                    raise
                    
    except Exception as e:
        logger.warning(f"Async orchestration failed: {e}. Falling back to sync.")
    
    # Fallback to synchronous orchestration
    logger.info("Using synchronous orchestration as fallback")
    return orchestrator.orchestrate(user_request)

def agent_orchestrator_dual_output(user_request: str) -> tuple:
    """Wrapper for OrchestratorAgent that returns both JSON and natural language output.
    Provides a unified interface to the orchestrator that returns structured data
    and a natural language summary of the orchestration process.
    Args:
        user_request (str): The user's request to be processed
    
    Returns:
            tuple: A tuple containing the orchestration result as a JSON dictionary
                   and a natural language summary of the process
    """
    result = orchestrator.orchestrate(user_request)
    
    # Extract the natural language summary from the result
    if isinstance(result, tuple) and len(result) > 0:
        json_result = result[0] if result[0] else {}
        
        # Create a natural language summary
        if isinstance(json_result, dict):
            summary = json_result.get('final_summary', '')
            if not summary:
                summary = json_result.get('summary', '')
            if not summary and 'code_output' in json_result:
                summary = f"Code executed successfully. Output: {json_result.get('code_output', {}).get('output', 'No output')}"
            if not summary:
                summary = "Process completed successfully."
        else:
            summary = "Process completed successfully."
    else:
        summary = "No results available."
        json_result = {}
    
    # Start warmup in background thread using the start_sandbox_warmup function
    start_sandbox_warmup()
    
    return json_result, summary

# ----------------------------------------
# Advanced Feature Functions
# ----------------------------------------

def get_health_status() -> Dict[str, Any]:
    """
    Get comprehensive system health status including advanced monitoring features.

    Retrieves detailed health information about the system including availability
    of advanced features, system resources, and operational metrics. Returns
    basic information if advanced monitoring is not available.

    Returns:
        Dict[str, Any]: A dictionary containing system health status and metrics
    """
    if not ADVANCED_FEATURES_AVAILABLE:
        return {
            "status": "basic_mode",
            "message": "Advanced features not available. Install 'pip install psutil aiohttp' to enable health monitoring.",
            "system_info": {
                "python_version": f"{types.__module__}",
                "gradio_available": True,
                "modal_available": True
            }
        }
    
    try:
        return health_monitor.get_health_stats()
    except Exception as e:
        return {"error": f"Health monitoring failed: {str(e)}"}

def get_performance_metrics() -> Dict[str, Any]:
    """
    Get performance metrics and analytics for the MCP Hub system.

    Collects and returns performance metrics including execution times,
    success rates, error counts, and resource utilization. Provides
    basic information if advanced metrics collection is not available.

    Returns:
        Dict[str, Any]: A dictionary containing performance metrics and statistics
    """
    if not ADVANCED_FEATURES_AVAILABLE:
        return {
            "status": "basic_mode", 
            "message": "Performance metrics not available. Install 'pip install psutil aiohttp' to enable advanced monitoring.",
            "basic_info": {
                "system_working": True,
                "features_loaded": False
            }
        }
    try:
        return metrics_collector.get_metrics_summary()
    except Exception as e:
        return {"error": f"Performance metrics failed: {str(e)}"}

def get_cache_status() -> Dict[str, Any]:
    """Get cache status and statistics."""
    if not ADVANCED_FEATURES_AVAILABLE:
        return {
            "status": "basic_mode",
            "message": "Cache monitoring not available. Install 'pip install psutil aiohttp' to enable cache statistics.",
            "cache_info": {
                "caching_available": False,
                "recommendation": "Install advanced features for intelligent caching"
            }
        }
    
    try:
        from mcp_hub.cache_utils import cache_manager
        return cache_manager.get_cache_status()
    except Exception as e:
        return {"error": f"Cache status failed: {str(e)}"}

async def get_sandbox_pool_status() -> Dict[str, Any]:
    """Get sandbox pool status and statistics."""
    try:
        # Create a temporary code runner to get pool stats
        code_runner = CodeRunnerAgent()
        stats = await code_runner.get_pool_stats()
        
        # Add warmup status information
        pool_size = stats.get("pool_size", 0)
        target_size = stats.get("target_pool_size", 0)
        
        if pool_size == 0:
            status_message = "🔄 Sandbox environment is warming up... This may take up to 2 minutes for the first execution."
            status = "warming_up"
        elif pool_size < target_size:
            status_message = f"⚡ Sandbox pool partially ready ({pool_size}/{target_size} sandboxes). More sandboxes warming up..."
            status = "partially_ready"
        else:
            status_message = f"✅ Sandbox pool fully ready ({pool_size}/{target_size} sandboxes available)"
            status = "ready"
        
        return {
            "status": status,
            "sandbox_pool": stats,
            "message": status_message,
            "user_message": status_message
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to get sandbox pool status: {str(e)}",
            "message": "Sandbox pool may not be initialized yet",
            "user_message": "🔄 Code execution environment is starting up... Please wait a moment."
        }

def get_sandbox_pool_status_sync() -> Dict[str, Any]:
    """Synchronous wrapper for sandbox pool status."""
    try:
        import asyncio
        return asyncio.run(get_sandbox_pool_status())
    except Exception as e:
        return {"error": f"Failed to get sandbox pool status: {str(e)}"}

def start_sandbox_warmup():
    """Start background sandbox warmup task."""
    try:
        import asyncio
        import threading
        
        def warmup_task():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Create a code runner to initialize the pool
                code_runner = CodeRunnerAgent()
                loop.run_until_complete(code_runner._ensure_pool_initialized())
                logger.info("Sandbox pool warmed up successfully")
            except Exception as e:
                logger.warning(f"Failed to warm up sandbox pool: {e}")
            finally:
                loop.close()
        
        # Start warmup in background thread
        warmup_thread = threading.Thread(target=warmup_task, daemon=True)
        warmup_thread.start()
        logger.info("Started background sandbox warmup")
        
    except Exception as e:
        logger.warning(f"Failed to start sandbox warmup: {e}")

class IntelligentCacheManager:
    """
    Advanced caching system for MCP Hub operations with TTL and eviction policies.

    Provides intelligent caching capabilities with time-to-live (TTL) support,
    automatic eviction of expired entries, and comprehensive cache statistics.
    Optimizes performance by caching operation results and managing memory usage.
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'total_requests': 0
        }
        self.max_cache_size = 1000
        self.default_ttl = 3600  # 1 hour        
    def _generate_cache_key(self, operation: str, **kwargs) -> str:
        """
        Generate a unique cache key based on operation and parameters.

        Creates a deterministic cache key by combining the operation name with
        parameter values. Uses MD5 hashing to ensure consistent key generation
        while keeping keys manageable in size.

        Args:
            operation (str): The operation name to include in the cache key
            **kwargs: Parameter values to include in the key generation

        Returns:
            str: A unique cache key as an MD5 hash string
        """
        import hashlib
        key_data = f"{operation}:{json.dumps(kwargs, sort_keys=True)}"        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, operation: str, **kwargs):
        """
        Retrieve cached data for a specific operation with automatic cleanup.

        Fetches cached data for the given operation and parameters. Automatically
        removes expired entries and updates cache statistics. Returns None if no
        valid cached data is found.

        Args:
            operation (str): The operation name to look up in cache
            **kwargs: Parameter values used to generate the cache key

        Returns:
            Any: The cached data if found and valid, otherwise None
        """
        cache_key = self._generate_cache_key(operation, **kwargs)
        self.cache_stats['total_requests'] += 1
        
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            current_time = time.time()
            
            if current_time < entry['expires_at']:
                self.cache_stats['hits'] += 1
                logger.info(f"Cache hit for operation: {operation}")
                return entry['data']
            else:
                # Remove expired entry
                del self.cache[cache_key]
        
        self.cache_stats['misses'] += 1
        return None
    
    def set(self, operation: str, data: Any, ttl: int = None, **kwargs):
        """Cache the result with TTL."""
        cache_key = self._generate_cache_key(operation, **kwargs)
        expires_at = time.time() + (ttl or self.default_ttl)
        
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_cache_size:
            self._evict_oldest_entries(int(self.max_cache_size * 0.1))
        
        self.cache[cache_key] = {
            'data': data,
            'expires_at': expires_at,
            'created_at': time.time()
        }
        logger.info(f"Cached result for operation: {operation}")
    
    def _evict_oldest_entries(self, count: int):
        """Remove the oldest entries from cache."""
        sorted_items = sorted(
            self.cache.items(),
            key=lambda x: x[1]['created_at']
        )
        for i in range(min(count, len(sorted_items))):
            del self.cache[sorted_items[i][0]]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        hit_rate = (self.cache_stats['hits'] / max(1, self.cache_stats['total_requests'])) * 100
        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.max_cache_size,
            'hit_rate': round(hit_rate, 2),
            'total_hits': self.cache_stats['hits'],
            'total_misses': self.cache_stats['misses'],
            'total_requests': self.cache_stats['total_requests']
        }
    
    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()
        logger.info("Cache cleared")


def agent_research_request(user_request):
    """
    This function researches a coding request from the user, generates code, executes it,
    and returns a clean summary of the results.

    This is an mcp server function that responds to research coding requests from users.

    Args:
        user_request (str): The user's request or question to be processed
    Returns:
        tuple: A tuple containing the JSON result from the orchestrator and a clean summary
    """
    # Get the full response (which is a tuple)
    orchestrator_result = agent_orchestrator(user_request)
    
    # Extract the JSON result (first element of tuple)
    if isinstance(orchestrator_result, tuple) and len(orchestrator_result) > 0:
        json_result = orchestrator_result[0]
    else:
        json_result = orchestrator_result
    
    # Extract and format the clean output
    clean_summary = ""
    if isinstance(json_result, dict):                
        if 'final_summary' in json_result:
            clean_summary += f"## 📋 Summary\n{json_result['final_summary']}\n\n"
        if 'code_string' in json_result and json_result['code_string']:
            clean_summary += f"## 💻 Generated Code\n```python\n{json_result['code_string']}\n```\n\n"
        
        if 'execution_output' in json_result and json_result['execution_output']:
            clean_summary += f"## ▶️ Execution Result\n```\n{json_result['execution_output']}\n```\n\n"
        
        if 'code_output' in json_result and json_result['code_output']:
            # Handle both string and dict formats for code_output
            code_output = json_result['code_output']
            if isinstance(code_output, dict):
                output = code_output.get('output', '')
            else:
                output = str(code_output)
            
            if output:
                clean_summary += f"## ▶️ Code Output\n```\n{output}\n```\n\n"
        
        if 'citations' in json_result and json_result['citations']:
            clean_summary += "## 📚 Sources\n"
            for i, citation in enumerate(json_result['citations'], 1):
                clean_summary += f"{i}. {citation}\n"
            clean_summary += "\n"
        
        if 'sub_questions' in json_result:
            clean_summary += "## 🔍 Research Questions Explored\n"
            for i, q in enumerate(json_result['sub_questions'], 1):
                clean_summary += f"{i}. {q}\n"
                
        # If we have sub-summaries, show them too
        if 'sub_summaries' in json_result and json_result['sub_summaries']:
            clean_summary += "\n## 📖 Research Summaries\n"
            for i, summary in enumerate(json_result['sub_summaries'], 1):
                clean_summary += f"### {i}. {summary}...\n"
    
    if not clean_summary:
        clean_summary = "## ⚠️ Processing Complete\nThe request was processed but no detailed results were generated."
    
    return json_result, clean_summary
# ----------------------------------------
# Gradio UI / MCP Server Setup
# ----------------------------------------

def agent_question_enhancer(user_request: str) -> dict:
    """
    Wrapper for QuestionEnhancerAgent to provide question enhancement.

    Args:
        user_request (str): The original user request to enhance

    Returns:
        dict: Enhanced question result with sub-questions
    """
    return question_enhancer.enhance_question(user_request, num_questions=2)

def agent_web_search(query: str) -> dict:
    """
    Wrapper for WebSearchAgent to perform web searches.

    Args:
        query (str): The search query to execute

    Returns:
        dict: Web search results with summaries and URLs
    """
    return web_search.search(query)

def agent_llm_processor(text_input: str, task: str, context: str | None = None) -> dict:
    """
    Wrapper for LLMProcessorAgent to process text with LLM.

    Args:
        text_input (str): The input text to process
        task (str): The processing task ('summarize', 'reason', or 'extract_keywords')
        context (str | None): Optional context for processing

    Returns:
        dict: LLM processing result with output and metadata
    """
    return llm_processor.process(text_input, task, context)

def agent_citation_formatter(text_block: str) -> dict:
    """
    Wrapper for CitationFormatterAgent to format citations.

    Args:
        text_block (str): The text containing URLs to cite

    Returns:
        dict: Formatted citations result with APA-style references
    """
    return citation_formatter.format_citations(text_block)

def agent_code_generator(user_request: str, grounded_context: str) -> tuple:
    """
    Wrapper for CodeGeneratorAgent to generate Python code.

    Args:
        user_request (str): The user's request for code generation
        grounded_context (str): Context information to guide generation

    Returns:
        tuple: A tuple containing the generation result and raw code
    """
    return code_generator.generate_code(user_request, grounded_context)

def code_runner_wrapper(code_or_obj) -> str:
    """
    Wrapper for CodeRunnerAgent that uses async execution with warm pool.

    Ensures a sandbox is spawned if not already present, waits for readiness,
    and then executes the code. Provides user-friendly error messages.

    Args:
        code_or_obj: The code string or object to be executed

    Returns:
        str: The execution result or user-friendly error message
    """
    try:
        import asyncio

        async def ensure_and_run():
            # Ensure the sandbox pool is initialized and ready
            await code_runner._ensure_pool_initialized()
            # Wait for at least one sandbox to be available
            pool_status = await get_sandbox_pool_status()
            user_message = pool_status.get("user_message", "")
            if pool_status.get("status") == "warming_up":
                return f"{user_message}\n\nPlease try again in a moment once the environment is ready."
            # Run the code in the sandbox
            return await code_runner.run_code_async(code_or_obj)

        return asyncio.run(ensure_and_run())

    except CodeExecutionError as e:
        error_msg = str(e)
        if "Failed to get sandbox" in error_msg or "timeout" in error_msg.lower():
            return (
                "🔄 The code execution environment is still starting up. Please wait a moment and try again.\n\n"
                "This is normal for the first execution after startup (can take 1-2 minutes)."
            )
        return error_msg
    except Exception as e:
        logger.error(f"Code runner wrapper error: {e}")
        return f"Error: {str(e)}"
    

def research_code(user_request: str) -> tuple:
    """
    This function serves as an MCP (Model Context Protocol) tool that orchestrates 
    comprehensive research and code generation workflows. It enhances user requests 
    through intelligent processing, performs web searches for relevant information, 
    generates appropriate code solutions, executes the code safely, and provides 
    clean, actionable summaries.
    The function is designed to be used as a tool within MCP frameworks, providing
    autonomous research capabilities that combine web search, code generation, and
    execution in a single workflow.
        user_request (str): The user's request, question, or problem statement to be 
                           processed. Can include coding problems, research questions, 
                           or requests for information gathering and analysis.
        tuple: A two-element tuple containing:
            - JSON result (dict): Structured data from the orchestrator containing 
              detailed research findings, generated code, execution results, and 
              metadata about the research process
            - Clean summary (str): A human-readable summary of the research findings 
              and generated solutions, formatted for easy consumption
    Example:
        >>> result, summary = research_code("How to implement a binary search in Python?")
        >>> print(summary)  # Clean explanation with code examples
        >>> print(result['code'])  # Generated code implementation
    Note:
        This function is optimized for use as an MCP tool and handles error cases
        gracefully, returning meaningful feedback even when research or code 
        generation encounters issues.
    """
    return agent_research_request(user_request)

CUSTOM_CSS = """
.app-title {
  text-align: center;
  font-family: 'Roboto', sans-serif;
  font-size: 3rem;
  font-weight: 700;
  letter-spacing: 1px;
  color: #10b981;
  text-shadow: 1px 1px 2px rgba(0,0,0,0.4);
  border-bottom: 4px solid #4f46e5;
  display: inline-block;
  padding-bottom: 0.5rem;
  margin: 2rem auto 1.5rem;
  max-width: 90%;
}
"""

# read the README.md file and convert it to a variable
with open("README.md", encoding="utf-8") as f:
    readme_content = f.read()


with gr.Blocks(title="Shallow Research Code Assistant Hub", 
               theme=gr.themes.Ocean(),
               fill_width=False,
               css=CUSTOM_CSS) as hub:
    
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                <h1 class="app-title" style="text-align: center; font-size: 2.5rem;">
                    Shallow Research Code Assistant Hub
                </h1>
                """,
                container=False,
            )

    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            gr.Markdown(
                """
                <h2>Welcome</h2>
                This hub provides a streamlined interface for AI-assisted research and code generation.
                It integrates multiple agents to enhance your coding and research workflow.

                The application can be accessed via the MCP server at:
                <code>https://agents-mcp-hackathon-shallowcoderesearch.hf.space/gradio_api/mcp/sse</code>
                <br></br>
                """,
                container=True,
                height=200,
            )

        with gr.Column(scale=1, min_width=320):
            gr.Image(
                value="static/CodeAssist.png",
                label="MCP Hub Logo",
                height=200,
                show_label=False,
                elem_id="mcp_hub_logo"
            )
        
    gr.Markdown(
        """
        <h3>Agents And Flows:</h3>
        """
    )
    with gr.Tab("README", scale=1):
        gr.Markdown(
            f"""{readme_content[371:]}
            """)
    
    with gr.Tab("Orchestrator Flow", scale=1):
        gr.Markdown("## AI Research & Code Assistant")
        gr.Markdown("""
        **Workflow:** Splits into two or more sub-questions → Tavily search & summarization → Generate Python code → Execute via Modal → Return results with citations
        """)
        
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                input_textbox = gr.Textbox(
                    label="Your High-Level Request", lines=12,
                    placeholder="Describe the code you need or the research topic you want to explore…",
                )
                process_btn = gr.Button("🚀 Process Request", variant="primary", size="lg")

                json_output = gr.JSON(label="Complete Orchestrated Output", 
                                      container=True,
                                      height=300,
                                      )
            with gr.Column(scale=1, min_width=300):
                with gr.Accordion("🔎 Show detailed summary", open=True):
                    clean_output = gr.Markdown(label="Summary & Results")

        process_btn.click(
            fn=agent_research_request,
            inputs=[input_textbox],
            outputs=[json_output, clean_output],
        )

    with gr.Tab("Agent: Question Enhancer", scale=1):
        gr.Interface(
            fn=agent_question_enhancer,
            inputs=[
                gr.Textbox(
                    label="Original User Request",
                    lines=12,
                    placeholder="Enter your question to be split into 3 sub-questions…"
                )
            ],
            outputs=gr.JSON(label="Enhanced Sub-Questions",
            height=305),
            title="Question Enhancer Agent",
            description="Splits a single user query into 3 distinct sub-questions using Qwen models.",
            api_name="agent_question_enhancer_service",
        )

    with gr.Tab("Agent: Web Search", scale=1):
        gr.Interface(
            fn=agent_web_search,
            inputs=[gr.Textbox(label="Search Query", placeholder="Enter search term…", lines=12)],
            outputs=gr.JSON(label="Web Search Results (Tavily)", height=305),
            title="Web Search Agent",
            description="Perform a Tavily web search with configurable result limits.",
            api_name="agent_web_search_service",
        )

    with gr.Tab("Agent: LLM Processor", scale=1):
        gr.Interface(
            fn=agent_llm_processor,
            inputs=[
                gr.Textbox(label="Text to Process", lines=12, placeholder="Enter text for the LLM…"),
                gr.Dropdown(
                    choices=["summarize", "reason", "extract_keywords"],
                    value="summarize",
                    label="LLM Task",
                ),
                gr.Textbox(label="Optional Context", lines=12, placeholder="Background info…"),
            ],
            outputs=gr.JSON(label="LLM Processed Output", height=1200),
            title="LLM Processing Agent",
            description="Use configured LLM provider for text processing tasks.",
            api_name="agent_llm_processor_service",
        )

    with gr.Tab("Agent: Citation Formatter", scale=1):
        gr.Interface(
            fn=agent_citation_formatter,
            inputs=[gr.Textbox(label="Text Block with Citations", lines=12, placeholder="Enter text to format citations…")],
            outputs=gr.JSON(label="Formatted Citations", height=305),
            title="Citation Formatter Agent",
            description="Extracts and formats APA-style citations from text blocks.",
            api_name="agent_citation_formatter_service",
        )
    with gr.Tab("Agent: Code Generator", scale=1):
        gr.Interface(
            fn=agent_code_generator,
            inputs=[
                gr.Textbox(label="User Request", lines=12, placeholder="Describe the code you need…"),
                gr.Textbox(label="Grounded Context", lines=12, placeholder="Context for code generation…")
            ],
            outputs=gr.JSON(label="Generated Code", height=610),
            title="Code Generation Agent",
            description="Generates Python code based on user requests and context.",
            api_name="agent_code_generator_service",
        )
    with gr.Tab("Agent: Code Runner", scale=1):
        gr.Interface(
            fn=code_runner_wrapper,
            inputs=[gr.Textbox(label="Code to Execute", lines=12, placeholder="Enter Python code to run…")],
            outputs=gr.Textbox(label="Execution Output", lines=12),
            title="Code Runner Agent",
            description="Executes Python code in a secure environment and returns the output.",
            api_name="agent_code_runner_service",
        )

    with gr.Tab("Advanced Features", scale=1):
        gr.Markdown("## Advanced Features")
        gr.Markdown("""
        **Available Features**:
        - **Health Monitoring**: System health and performance metrics.
        - **Performance Analytics**: Detailed performance statistics.
        - **Intelligent Caching**: Advanced caching system for improved efficiency.
        - **Sandbox Pool Status**: Monitor warm sandbox pool performance and statistics.
        
        **Note**: Some features require additional dependencies. Install with `pip install psutil aiohttp` to enable all features.
        """)
        
        with gr.Row():
            health_btn = gr.Button("Get Health Status", variant="primary")
            metrics_btn = gr.Button("Get Performance Metrics", variant="primary")
            cache_btn = gr.Button("Get Cache Status", variant="primary")
            sandbox_btn = gr.Button("Get Sandbox Pool Status", variant="primary")
        
        health_output = gr.JSON(label="Health Status")
        metrics_output = gr.JSON(label="Performance Metrics")
        cache_output = gr.JSON(label="Cache Status")
        sandbox_output = gr.JSON(label="Sandbox Pool Status")
        
        health_btn.click(
            fn=get_health_status,
            inputs=[],
            outputs=health_output,
            api_name="get_health_status_service"
        )
        
        metrics_btn.click(
            fn=get_performance_metrics,
            inputs=[],
            outputs=metrics_output,
            api_name="get_performance_metrics_service"
        )
        
        cache_btn.click(
            fn=get_cache_status,
            inputs=[],
            outputs=cache_output,
            api_name="get_cache_status_service"
        )
        
        sandbox_btn.click(
            fn=get_sandbox_pool_status_sync,
            inputs=[],
            outputs=sandbox_output,
            api_name="get_sandbox_pool_status_service"
        )

# ----------------------------------------
# Main Entry Point
# ----------------------------------------
if __name__ == "__main__":
    import signal
    import atexit
    
    # Start the background warmup task for sandbox pool
    start_sandbox_warmup()
    
    # Register cleanup functions for graceful shutdown
    def cleanup_on_exit():
        """Cleanup function to run on exit."""
        try:
            import asyncio
            
            # Attempt to cleanup sandbox pool
            def run_cleanup():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    code_runner = CodeRunnerAgent()
                    if code_runner._pool_initialized:
                        loop.run_until_complete(code_runner.cleanup_pool())
                        logger.info("Sandbox pool cleaned up on exit")
                except Exception as e:
                    logger.warning(f"Failed to cleanup sandbox pool on exit: {e}")
                finally:
                    loop.close()
            
            run_cleanup()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    # Register cleanup handlers
    atexit.register(cleanup_on_exit)
    
    def signal_handler(signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating cleanup...")
        cleanup_on_exit()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler) 
    
    try:
        hub.launch(
            mcp_server=True,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            share=True
        )
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        cleanup_on_exit()
    except Exception as e:
        logger.error(f"Application error: {e}")
        cleanup_on_exit()
        raise

