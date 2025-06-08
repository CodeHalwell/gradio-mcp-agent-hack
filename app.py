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

# Import agent classes
from mcp_hub.agents import (
    QuestionEnhancerAgent,
    WebSearchAgent,
    LLMProcessorAgent,
    CitationFormatterAgent,
    CodeGeneratorAgent,
    CodeRunnerAgent,
    OrchestratorAgent
)

# Import orchestrator utility functions
from mcp_hub.agents.orchestrator import (
    agent_orchestrator,
    agent_orchestrator_dual_output,
    get_health_status,
    get_performance_metrics,
    get_cache_status,
    get_sandbox_pool_status,
    get_sandbox_pool_status_sync,
    start_sandbox_warmup
)

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

    Provides a simplified interface to the code runner with automatic sandbox
    pool management and user-friendly error messages. Handles warm-up status
    checks and provides appropriate feedback during startup.

    Args:
        code_or_obj: The code string or object to be executed

    Returns:
        str: The execution result or user-friendly error message
    """
    try:
        import asyncio
        
        # First check sandbox pool status to provide user feedback        
        try:
            pool_status = asyncio.run(get_sandbox_pool_status())
            user_message = pool_status.get("user_message", "")
            if pool_status.get("status") == "warming_up":
                return f"{user_message}\n\nPlease try again in a moment once the environment is ready."
        except Exception:
            pass  # Continue with execution even if status check fails
        
        # Use async execution to leverage the warm sandbox pool
        result = asyncio.run(code_runner.run_code_async(code_or_obj))
        return result
    except CodeExecutionError as e:
        error_msg = str(e)
        if "Failed to get sandbox" in error_msg or "timeout" in error_msg.lower():
            return "🔄 The code execution environment is still starting up. Please wait a moment and try again.\n\nThis is normal for the first execution after startup (can take 1-2 minutes)."
        return error_msg
    except Exception as e:
        logger.error(f"Code runner wrapper error: {e}")
        return f"Error: {str(e)}"

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


with gr.Blocks(title="Shallow Research Code Assistant Hub", 
               theme=gr.themes.Ocean(),
               fill_width=False,
               css=CUSTOM_CSS) as demo:
    
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
                <code></code>
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

        def validate_and_process(user_request):
            if not user_request or user_request.strip() == "":
                return None, "Please enter a request before processing."
            return process_orchestrator_request(user_request)

        process_btn.click(
            fn=validate_and_process,
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
        demo.launch(
            mcp_server=True,
            server_name="127.0.0.1",
            server_port=7860,
            show_error=True,
            share=False
        )
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        cleanup_on_exit()
    except Exception as e:
        logger.error(f"Application error: {e}")
        cleanup_on_exit()
        raise

