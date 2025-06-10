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

class QuestionEnhancerAgent:
    """
    Agent responsible for enhancing questions into sub-questions for research.

    This agent takes a single user query and intelligently breaks it down into
    multiple distinct, non-overlapping sub-questions that explore different
    technical angles of the original request. It uses LLM models to enhance
    question comprehension and research depth.    """
    
    @with_performance_tracking("question_enhancement")
    @rate_limited("nebius")
    @circuit_protected("nebius")
    @cached(ttl=300)  # Cache for 5 minutes
    def enhance_question(self, user_request: str, num_questions: int) -> Dict[str, Any]:
        """
        Split a single user query into multiple distinct sub-questions for enhanced research.

        Takes a user's original request and uses LLM processing to break it down into
        separate sub-questions that explore different technical angles. This enables
        more comprehensive research and analysis of complex topics.

        Args:
            user_request (str): The original user query to be enhanced and split
            num_questions (int): The number of sub-questions to generate

        Returns:
            Dict[str, Any]: A dictionary containing the generated sub-questions array
                           or error information if processing fails
        """
        try:
            validate_non_empty_string(user_request, "User request")
            logger.info(f"Enhancing question: {user_request[:100]}...")
            
            prompt_text = f"""
            You are an AI assistant specialised in Python programming that must break a single user query into {num_questions} distinct, non-overlapping sub-questions.
            Each sub-question should explore a different technical angle of the original request.
            Output must be valid JSON with a top-level key "sub_questions" whose value is an array of strings—no extra keys, no extra prose.

            User Request: "{user_request}"

            Respond with exactly:
            {{
            "sub_questions": [
                "First enhanced sub-question …",
                "Second enhanced sub-question …",
                ........ more added as necessary
            ]
            }}
            """
            
            messages = [{"role": "user", "content": prompt_text}]
            response_format = {
                "type": "json_object",
                "object": {
                    "sub_questions": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
            }

            logger.info(
                "The LLM provider is: %s and the model is: %s",
                api_config.llm_provider,
                model_config.get_model_for_provider("question_enhancer", api_config.llm_provider)
            )
            
            raw_output = make_llm_completion(
                model=model_config.get_model_for_provider("question_enhancer", api_config.llm_provider),
                messages=messages,
                temperature=0.7,
                response_format=response_format
            )
            
            parsed = extract_json_from_text(raw_output)
            
            if "sub_questions" not in parsed:
                raise ValidationError("JSON does not contain a 'sub_questions' key.")
            
            sub_questions = parsed["sub_questions"]
            if not isinstance(sub_questions, list) or not all(isinstance(q, str) for q in sub_questions):
                raise ValidationError("Expected 'sub_questions' to be a list of strings.")
            
            logger.info(f"Successfully generated {len(sub_questions)} sub-questions")
            return {"sub_questions": sub_questions}
            
        except (ValidationError, APIError) as e:
            logger.error(f"Question enhancement failed: {str(e)}")
            return {"error": str(e), "sub_questions": []}
        except Exception as e:
            logger.error(f"Unexpected error in question enhancement: {str(e)}")
            return {"error": f"Unexpected error: {str(e)}", "sub_questions": []}

class WebSearchAgent:
    """
    Agent responsible for performing web searches using the Tavily API.

    This agent handles web search operations to gather information from the internet.
    It provides both synchronous and asynchronous search capabilities with configurable
    result limits and search depth. Results include summaries, URLs, and content snippets.
    """
    
    def __init__(self):
        if not api_config.tavily_api_key:
            raise APIError("Tavily", "API key not configured")
        self.client = TavilyClient(api_key=api_config.tavily_api_key)
    
    @with_performance_tracking("web_search")
    @rate_limited("tavily")
    @circuit_protected("tavily")
    @cached(ttl=600)  # Cache for 10 minutes
    def search(self, query: str) -> Dict[str, Any]:
        """
        Perform a web search using the Tavily API to gather internet information.

        Executes a synchronous web search with the specified query and returns
        structured results including search summaries, URLs, and content snippets.
        Results are cached for performance optimization.

        Args:
            query (str): The search query string to look up on the web

        Returns:
            Dict[str, Any]: A dictionary containing search results, summaries, and metadata
                           or error information if the search fails
        """
        try:
            validate_non_empty_string(query, "Search query")
            logger.info(f"Performing web search: {query}")
            
            response = self.client.search(
                query=query,
                search_depth="basic",
                max_results=app_config.max_search_results,
                include_answer=True
            )
            
            logger.info(f"Search completed, found {len(response.get('results', []))} results")
            return {
                "query": response.get("query", query),
                "tavily_answer": response.get("answer"),
                "results": response.get("results", []),
                "data_source": "Tavily Search API",
            }
            
        except ValidationError as e:
            logger.error(f"Web search validation failed: {str(e)}")
            return {"error": str(e), "query": query, "results": []}
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            return {"error": f"Tavily API Error: {str(e)}", "query": query, "results": []}
    
    @with_performance_tracking("async_web_search")
    @rate_limited("tavily")
    @circuit_protected("tavily")
    async def search_async(self, query: str) -> Dict[str, Any]:
        """
        Perform an asynchronous web search using aiohttp for better performance.

        Executes an async web search with the specified query using direct HTTP calls
        to the Tavily API. Falls back to synchronous search if async fails.
        Provides better performance for concurrent operations.

        Args:
            query (str): The search query string to look up on the web

        Returns:
            Dict[str, Any]: A dictionary containing search results, summaries, and metadata
                           or falls back to synchronous search on error
        """
        try:
            validate_non_empty_string(query, "Search query")
            logger.info(f"Performing async web search: {query}")
            
            # Use async HTTP client for better performance
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {api_config.tavily_api_key}',
                    'Content-Type': 'application/json'
                }
                
                payload = {
                    'query': query,
                    'search_depth': 'basic',
                    'max_results': app_config.max_search_results,
                    'include_answer': True
                }
                
                async with session.post(
                    'https://api.tavily.com/search',
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Async search completed, found {len(data.get('results', []))} results")
                        return {
                            "query": data.get("query", query),
                            "tavily_answer": data.get("answer"),
                            "results": data.get("results", []),
                            "data_source": "Tavily Search API (Async)",
                        }
                    else:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")
            
        except ValidationError as e:
            logger.error(f"Async web search validation failed: {str(e)}")
            return {"error": str(e), "query": query, "results": []}
        except Exception as e:
            logger.error(f"Async web search failed: {str(e)}")
            # Fallback to sync version on error
            logger.info("Falling back to synchronous search")
            return self.search(query)

class LLMProcessorAgent:
    """
    Agent responsible for processing text using Large Language Models for various tasks.

    This agent handles text processing operations including summarization, reasoning,
    and keyword extraction using configured LLM providers. It supports both synchronous
    and asynchronous processing with configurable temperature and response formats.    """
    
    @with_performance_tracking("llm_processing")
    @rate_limited("nebius")
    @circuit_protected("nebius")
    def process(self, text_input: str, task: str, context: str = None) -> Dict[str, Any]:
        """
        Process text using LLM for summarization, reasoning, or keyword extraction.

        Applies the configured LLM model to process the input text according to the
        specified task type. Supports summarization for condensing content, reasoning
        for analytical tasks, and keyword extraction for identifying key terms.

        Args:
            text_input (str): The input text to be processed by the LLM
            task (str): The processing task ('summarize', 'reason', or 'extract_keywords')
            context (str, optional): Additional context to guide the processing

        Returns:
            Dict[str, Any]: A dictionary containing the processed output and metadata
                           or error information if processing fails
        """
        try:
            validate_non_empty_string(text_input, "Input text")
            validate_non_empty_string(task, "Task")
            logger.info(f"Processing text with task: {task}")
            
            task_lower = task.lower()
            if task_lower not in ["reason", "summarize", "extract_keywords"]:
                raise ValidationError(
                    f"Unsupported LLM task: {task}. Choose 'summarize', 'reason', or 'extract_keywords'."
                )
            
            prompt_text = self._build_prompt(text_input, task_lower, context)
            messages = [{"role": "user", "content": prompt_text}]

            logger.info(f"LLM provider is: {api_config.llm_provider}, model used: {model_config.get_model_for_provider('llm_processor', api_config.llm_provider)}")
            
            output_text = make_llm_completion(
                model=model_config.get_model_for_provider("llm_processor", api_config.llm_provider),
                messages=messages,
                temperature=app_config.llm_temperature
            )
            
            logger.info(f"LLM processing completed for task: {task}")
            return {
                "input_text": text_input,
                "task": task,
                "provided_context": context,
                "llm_processed_output": output_text,
                "llm_model_used": model_config.get_model_for_provider("llm_processor", api_config.llm_provider),
            }
            
        except (ValidationError, APIError) as e:
            logger.error(f"LLM processing failed: {str(e)}")
            return {"error": str(e), "input_text": text_input, "processed_output": None}
        except Exception as e:
            logger.error(f"Unexpected error in LLM processing: {str(e)}")
            return {"error": f"Unexpected error: {str(e)}", "input_text": text_input, "processed_output": None}

    @with_performance_tracking("async_llm_processing")
    @rate_limited("nebius")
    @circuit_protected("nebius")
    async def async_process(self, text_input: str, task: str, context: str = None) -> Dict[str, Any]:
        """
        Process text using async LLM for summarization, reasoning, or keyword extraction.

        Asynchronous version of the text processing function that provides better
        performance for concurrent operations. Uses async LLM completion calls
        for improved throughput when processing multiple texts simultaneously.

        Args:
            text_input (str): The input text to be processed by the LLM
            task (str): The processing task ('summarize', 'reason', or 'extract_keywords')
            context (str, optional): Additional context to guide the processing

        Returns:
            Dict[str, Any]: A dictionary containing the processed output and metadata
                           or error information if processing fails
        """
        try:
            validate_non_empty_string(text_input, "Input text")
            validate_non_empty_string(task, "Task")
            logger.info(f"Processing text async with task: {task}")
            
            task_lower = task.lower()
            if task_lower not in ["reason", "summarize", "extract_keywords"]:
                raise ValidationError(
                    f"Unsupported LLM task: {task}. Choose 'summarize', 'reason', or 'extract_keywords'."
                )
            
            prompt_text = self._build_prompt(text_input, task_lower, context)
            messages = [{"role": "user", "content": prompt_text}]

            logger.info(f"LLM provider is: {api_config.llm_provider}, model used: {model_config.get_model_for_provider('llm_processor', api_config.llm_provider)}")
            
            from mcp_hub.utils import make_async_llm_completion
            output_text = await make_async_llm_completion(
                model=model_config.get_model_for_provider("llm_processor", api_config.llm_provider),
                messages=messages,
                temperature=app_config.llm_temperature
            )
            
            logger.info(f"Async LLM processing completed for task: {task}")
            return {
                "input_text": text_input,
                "task": task,
                "provided_context": context,
                "llm_processed_output": output_text,
                "llm_model_used": model_config.get_model_for_provider("llm_processor", api_config.llm_provider),
            }
            
        except (ValidationError, APIError) as e:
            logger.error(f"Async LLM processing failed: {str(e)}")
            return {"error": str(e), "input_text": text_input, "processed_output": None}
        except Exception as e:
            logger.error(f"Unexpected error in async LLM processing: {str(e)}")
            return {"error": f"Unexpected error: {str(e)}", "input_text": text_input, "processed_output": None}
    
    def _build_prompt(self, text_input: str, task: str, context: str = None) -> str:
        """Build the appropriate prompt based on the task."""
        prompts = {
            "reason": f"Analyze this text and provide detailed reasoning (less than 250):\n\n{text_input} with this context {context if context else ''} for {task}",
            "summarize": f"Summarize in detail (less than 250):\n\n{text_input} with this context {context if context else ''} for {task}",
            "extract_keywords": f"Extract key terms/entities (comma-separated) from:\n\n{text_input}"
        }
        
        prompt = prompts[task]
        
        if context:
            context_additions = {
                "reason": f"\n\nAdditional context: {context}",
                "summarize": f"\n\nKeep in mind this context: {context}",
                "extract_keywords": f"\n\nFocus on this context: {context}"
            }
            prompt += context_additions[task]
        
        task_endings = {
            "reason": "\n\nReasoning:",
            "summarize": "\n\nSummary:",
            "extract_keywords": "\n\nKeywords:"
        }
        prompt += task_endings[task]
        
        return prompt

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

class CodeGeneratorAgent:
    """
    Agent responsible for generating Python code based on user requests and context.

    This agent generates secure Python code using LLM models with built-in security
    checks and validation. It enforces restrictions on dangerous function calls and
    modules, ensures code compilation, and provides iterative error correction.
    """

    # List of disallowed function calls for security    
    DISALLOWED_CALLS = {
        "input", "eval", "exec", "compile", "__import__", "open", 
        "file", "raw_input", "execfile", "reload", "quit", "exit"
    }
    
    def _uses_disallowed_calls(self, code_str: str) -> tuple[bool, list[str]]:
        """Check if code uses disallowed function calls."""
        violations = []
        try:
            tree = ast.parse(code_str)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in self.DISALLOWED_CALLS:
                        violations.append(node.func.id)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ["os", "subprocess", "sys"]:
                            violations.append(f"import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module in ["os", "subprocess", "sys"]:
                        violations.append(f"from {node.module} import ...")        
        except SyntaxError:
            # Don't treat syntax errors as security violations - let them be handled separately
            return False, []
        
        return len(violations) > 0, violations

    def _make_prompt(self, user_req: str, ctx: str, prev_err: str = "") -> str:
        """Create a prompt for code generation with error feedback."""
        disallowed_list = ", ".join(self.DISALLOWED_CALLS)
        prev_error_text = ""
        if prev_err:
            prev_error_text = f"Previous attempt failed:\n{prev_err}\nFix it."
        
        return f"""
                You are an expert Python developer. **Rules**:
                - Never use these functions: {disallowed_list}
                - Never import os, subprocess, or sys modules
                - After defining functions/classes, call them and print the result.
                - Always include print statements to show output
                {prev_error_text}

                USER REQUEST:
                \"\"\"{user_req}\"\"\"

                CONTEXT:
                \"\"\"{ctx}\"\"\"

                Provide only valid Python code that can be executed safely.

                Provide only the Python code and never under any circumstance include any
                explanations in your response. **Do not include back ticks or the word python
                and dont include input fields**

                for example,

                import requests
                response = requests.get("https://api.example.com/data")
                print(response.json())

                or

                def add_numbers(a, b):
                    return a + b
                result = add_numbers(5, 10)
                print(result)

                NEVER include input() or Never use input(), even in disguised forms like raw_input()

                ALWAYS return valid Python code that can be executed without errors. The code returned should be
                a function or class depending on the complexity. For simple requests, return a function, 
                and for more complex requests, return a class with methods that can be called.

                After the creation of classes or functions, classes should be instantiated or functions should be called
                to demonstrate their usage. The final step is include the print function of the result of the class and/or function.

                for example

                class DataFetcher:
                def __init__(self, url):
                    self.url = url
                def fetch_data(self):
                    response = requests.get(self.url)
                    return response.json()
                fetcher = DataFetcher("https://api.example.com/data")
                data = fetcher.fetch_data()
                print(data)

                if the code requires and data manipulation etc, generate the code to test the code and print the result.

                for example;
                def process_data(data):
                    # Perform some data manipulation
                    return data * 2
                data = 5

                or 

                For example, to get the mean of a column in a pandas DataFrame:

                import pandas as pd

                def get_mean_of_column(df, column_name):
                    return df[column_name].mean()

                df = pd.DataFrame({{'A': [1, 2, 3], 'B': [4, 5, 6]}})
                mean_value = get_mean_of_column(df, 'A')
                print(mean_value)

                # If you want to pretty-print the DataFrame:
                import json
                print(json.dumps(df.to_dict(), indent=2))

                Never wrap dictionaries or lists in f-strings in print statements (e.g., avoid print(f"{{my_dict}}")).

                To print a dict or list, use print(my_dict) or, if you want pretty output, use the json module:

                import json
                print(json.dumps(my_dict, indent=2))
                If you need to include a variable in a string, only use f-strings with simple values, not dicts or lists.


                
                Never wrap dictionaries or lists in f-strings in print statements, like this:

                # ❌ BAD EXAMPLE — NEVER DO THIS:
                my_dict = {{'A': [1,2,3], 'B': [4,5,6]}}
                print(f"{{my_dict}}")

                # ❌ BAD EXAMPLE — NEVER DO THIS:
                my_list = [1, 2, 3]
                print(f"{{my_list}}")

                # ✅ GOOD EXAMPLES — ALWAYS DO THIS INSTEAD:
                print(my_dict)
                print(my_list)

                # ✅ Or, for pretty output, do:
                import json
                print(json.dumps(my_dict, indent=2))

                If you need to include a variable in a string, only use f-strings with simple scalar values, not dicts or lists. For example:

                # ✅ Good f-string with a simple value:
                mean = 3.5
                print(f"The mean is {{mean}}")

                # ❌ Bad f-string with a dict:
                print(f"The data is {{my_dict}}")   # <-- NEVER DO THIS

                # ✅ Good way to show a dict:
                print("The data is:", my_dict)

                Generated code like this is stricly forbidden due to the word python and the backticks
                ```python
                import x
                import y
                def my_function(i):
                    return i + 1
                ```

                ### **Summary**

                - Repeat the "NEVER wrap dicts/lists in f-strings" rule.
                - Use all-caps or bold/emoji to make "NEVER" and "ALWAYS" pop out.
                - Finish the prompt by *repeating* the most important style rule.
                - **NEVER** include backticks like this ` or the word "python" in the response.
                - Return **ONLY** the actual code as a string without any additional text.
                """

    @with_performance_tracking("code_generation")
    @rate_limited("nebius")    
    @circuit_protected("nebius")
    def generate_code(
        self, user_request: str, grounded_context: str
    ) -> tuple[Dict[str, Any], str]:
        """
        Generate Python code based on user request and grounded context with enhanced security.

        Creates safe, executable Python code using LLM models with built-in security
        validation. Includes iterative error correction, syntax checking, and
        security violation detection to ensure safe code generation.

        Args:
            user_request (str): The user's request describing what code to generate
            grounded_context (str): Contextual information to inform code generation

        Returns:
            tuple[Dict[str, Any], str]: A tuple containing the generation result dictionary
                                       and the raw generated code string
        """
        try:
            validate_non_empty_string(user_request, "User request")
            logger.info("Generating Python code with security checks")

            prev_error = ""
            
            for attempt in range(1, app_config.max_code_generation_attempts + 1):
                try:
                    logger.info(f"Code generation attempt {attempt}")

                    prompt_text = self._make_prompt(user_request, grounded_context, prev_error)
                    messages = [{"role": "user", "content": prompt_text}]
                    
                    logger.info(f"LLM provider is: {api_config.llm_provider}, model used: {model_config.get_model_for_provider('code_generator', api_config.llm_provider)}")

                    raw_output = make_llm_completion(
                        model=model_config.get_model_for_provider("code_generator", api_config.llm_provider),
                        messages=messages,
                        temperature=app_config.code_gen_temperature,
                    )
                    logger.info(f"Generated code (attempt {attempt}):\n{raw_output}\n")
                    
                    # First, validate that the code compiles (syntax check)
                    try:
                        code_compiled = compile(raw_output, "<string>", "exec")
                    except SyntaxError as syntax_err:
                        prev_error = f"Syntax error: {str(syntax_err)}"
                        logger.warning(f"Generated code syntax error (attempt {attempt}): {syntax_err}")
                        if attempt == app_config.max_code_generation_attempts:
                            raise CodeGenerationError(
                                f"Failed to generate valid Python syntax after {attempt} attempts"
                            )
                        continue
                    
                    # Then security check: look for disallowed calls (only if syntax is valid)
                    has_violations, violations = self._uses_disallowed_calls(raw_output)
                    if has_violations:
                        prev_error = f"Security violation - used disallowed functions: {', '.join(violations)}"
                        logger.warning(f"Security violation in attempt {attempt}: {violations}")
                        if attempt == app_config.max_code_generation_attempts:
                            raise CodeGenerationError(f"Code contains security violations: {violations}")
                        continue

                    logger.info(f"The generated code is as follows: \n\n{raw_output}\n")
                    logger.info("Code generation successful with security checks passed")

                    return {"status": "success", "generated_code": code_compiled, "code": code_compiled}, raw_output

                except SyntaxError as e:
                    prev_error = f"Syntax error: {str(e)}"
                    logger.warning(f"Generated code syntax error (attempt {attempt}): {e}")
                    if attempt == app_config.max_code_generation_attempts:
                        raise CodeGenerationError(
                            f"Failed to generate valid Python after {attempt} attempts"
                        )
                    continue

                except APIError as e:
                    raise CodeGenerationError(f"Unexpected API error: {e}") from e

                except Exception as e:
                    prev_error = f"Unexpected error: {str(e)}"
                    logger.error(f"Code generation error (attempt {attempt}): {e}")
                    if attempt == app_config.max_code_generation_attempts:
                        raise CodeGenerationError(f"Unexpected error: {e}")
                    continue

            raise CodeGenerationError("No valid code produced after all attempts")        
        except (ValidationError, APIError, CodeGenerationError) as e:
            logger.error("Code generation failed: %s", e)
            return {"error": str(e), "generated_code": ""}, ""
            
        except Exception as e:
            logger.error("Unexpected error in code generation: %s", e)
            return {"error": f"Unexpected error: {e}", "generated_code": ""}, ""

    
    def _get_enhanced_image(self):
        """Get Modal image with enhanced security and performance packages."""
        return (
            modal.Image.debian_slim(python_version="3.12")
            .pip_install([
                "numpy", "pandas", "matplotlib", "seaborn", "plotly",
                "requests", "beautifulsoup4", "lxml", "scipy", "scikit-learn",
                "pillow", "opencv-python-headless", "wordcloud", "textblob"
            ])
            .apt_install(["curl", "wget", "git"])
            .env({"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"})
            .run_commands([
                "python -m pip install --upgrade pip",
                "pip install --no-cache-dir jupyter ipython"
            ])
        )

class CodeRunnerAgent:
    """
    Agent responsible for executing code in Modal sandbox with enhanced security.

    This agent provides secure code execution in isolated Modal sandbox environments
    with warm sandbox pools for performance optimization. It includes safety shims,
    package management, and both synchronous and asynchronous execution capabilities.
    """
    
    def __init__(self):
        self.app = modal.App.lookup(app_config.modal_app_name, create_if_missing=True)
        # Create enhanced image with common packages for better performance
        self.image = self._create_enhanced_image()
        # Initialize warm sandbox pool
        self.sandbox_pool = None
        self._pool_initialized = False
    
    def _create_enhanced_image(self):
        """Create a lean Modal image with only essential packages pre-installed."""
        # Only include truly essential packages in the base image to reduce cold start time
        essential_packages = [
            "numpy",
            "pandas", 
            "matplotlib",
            "requests",
            "scikit-learn",
        ]
        
        try:
            return (
                modal.Image.debian_slim()
                .pip_install(*essential_packages)
                .apt_install(["curl", "wget", "git"])
                .env({"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"})
            )
        except Exception as e:
            logger.warning(f"Failed to create enhanced image, using basic: {e}")
            return modal.Image.debian_slim()
    
    async def _ensure_pool_initialized(self):
        """Ensure the sandbox pool is initialized (lazy initialization)."""
        if not self._pool_initialized:
            from mcp_hub.sandbox_pool import WarmSandboxPool
            self.sandbox_pool = WarmSandboxPool(
                app=self.app,
                image=self.image,
                pool_size=5,  # Increased from 3 to reduce cold starts
                max_age_seconds=600,  # Increased from 300 (10 minutes)
                max_uses_per_sandbox=10
            )
            await self.sandbox_pool.start()
            self._pool_initialized = True
            logger.info("Warm sandbox pool initialized")
    
    async def get_pool_stats(self):
        """Get sandbox pool statistics."""
        if self.sandbox_pool:
            return self.sandbox_pool.get_stats()
        return {"error": "Pool not initialized"}
    
    @asynccontextmanager
    async def _sandbox_context(self, **kwargs):
        """Context manager for safe sandbox lifecycle management."""
        sb = None
        try:
            sb = modal.Sandbox.create(
                app=self.app, 
                image=self.image,
                cpu=1.0,
                memory=512,  # MB
                timeout=30,  # seconds
                **kwargs
            )
            yield sb
        except Exception as e:
            logger.error(f"Sandbox creation failed: {e}")
            raise CodeExecutionError(f"Failed to create sandbox: {e}")
        finally:
            if sb:                
                try:
                    sb.terminate()
                except Exception as e:
                    logger.warning(f"Failed to terminate sandbox: {e}")

    def _add_safety_shim(self, code: str) -> str:
        """Return code wrapped in the security shim, for file-based execution."""
        try:
            safety_shim = f"""
import sys
import types
import functools
import builtins
import marshal
import traceback

RESTRICTED_BUILTINS = {{
    'open', 'input', 'eval', 'compile', '__import__',
    'getattr', 'setattr', 'delattr', 'hasattr', 'globals', 'locals',
    'pty', 'subprocess', 'socket', 'threading', 'ssl', 'email', 'smtpd'
}}

if isinstance(__builtins__, dict):
    _original_builtins = __builtins__.copy()
else:
    _original_builtins = __builtins__.__dict__.copy()

_safe_builtins = {{k: v for k, v in _original_builtins.items() if k not in RESTRICTED_BUILTINS}}
_safe_builtins['print'] = print

def safe_exec(code_obj, globals_dict=None, locals_dict=None):
    if not isinstance(code_obj, types.CodeType):
        raise TypeError("safe_exec only accepts a compiled code object")
    if globals_dict is None:
        globals_dict = {{"__builtins__": types.MappingProxyType(_safe_builtins)}}
    return _original_builtins['exec'](code_obj, globals_dict, locals_dict)

_safe_builtins['exec'] = safe_exec

def safe_import(name, *args, **kwargs):
    ALLOWED_MODULES = (
        set(sys.stdlib_module_names)
        .difference(RESTRICTED_BUILTINS)
        .union({{
    "aiokafka", "altair", "anthropic", "apache-airflow", "apsw", "bokeh", "black", "bottle", "catboost", "click",
    "confluent-kafka", "cryptography", "cupy", "dask", "dash", "datasets", "dagster", "django", "distributed", "duckdb",
    "duckdb-engine", "elasticsearch", "evidently", "fastapi", "fastparquet", "flake8", "flask", "folium", "geopandas", "geopy",
    "gensim", "google-cloud-aiplatform", "google-cloud-bigquery", "google-cloud-pubsub", "google-cloud-speech", "google-cloud-storage",
    "google-cloud-texttospeech", "google-cloud-translate", "google-cloud-vision", "google-genai", "great-expectations", "holoviews",
    "html5lib", "httpx", "huggingface_hub", "hvplot", "imbalanced-learn", "imageio", "isort", "jax", "jaxlib",
    "jsonschema",  # added for data validation
    "langchain", "langchain_aws", "langchain_aws_bedrock", "langchain_aws_dynamodb", "langchain_aws_lambda", "langchain_aws_s3",
    "langchain_aws_sagemaker", "langchain_azure", "langchain_azure_openai", "langchain_chroma", "langchain_community",
    "langchain_core", "langchain_elasticsearch", "langchain_google_vertex", "langchain_huggingface", "langchain_mongodb",
    "langchain_openai", "langchain_ollama", "langchain_pinecone", "langchain_redis", "langchain_sqlalchemy",
    "langchain_text_splitters", "langchain_weaviate", "lightgbm", "llama-cpp-python", "lxml", "matplotlib", "mlflow", "modal", "mypy",
    "mysql-connector-python", "networkx", "neuralprophet", "nltk", "numba", "numpy", "openai", "opencv-python", "optuna", "panel",
    "pandas", "pendulum", "poetry", "polars", "prefect", "prophet", "psycopg2", "pillow", "pyarrow", "pydeck",
    "pyjwt", "pylint", "pymongo", "pymupdf", "pyproj", "pypdf", "pypdf2", "pytest", "python-dateutil", "pytorch-lightning",
    "ray", "ragas", "rapidsai-cuda11x",  # optional: GPU dataframe ops
    "redis", "reportlab", "requests", "rich", "ruff", "schedule", "scikit-image", "scikit-learn", "scrapy", "scipy",
    "seaborn", "sentence-transformers", "shap", "shapely", "sqlite-web", "sqlalchemy", "starlette", "statsmodels", "streamlit",
    "sympy", "tensorflow", "torch", "transformers", "tqdm", "typer", "vllm", "wandb", "watchdog", "xgboost",
}})
    )
    if name in ALLOWED_MODULES:
        return _original_builtins['__import__'](name, *args, **kwargs)
    raise ImportError(f"Module {{name!r}} is not allowed in this environment")

_safe_builtins['__import__'] = safe_import

try:
{self._indent_code(code)}
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    traceback.print_exc()
"""
            return safety_shim
        except Exception as e:
            logger.error(f"Failed to add safety shim: {str(e)}")
            raise CodeExecutionError(f"Failed to prepare safe code execution: {str(e)}")

    def _indent_code(self, code: str, indent: int = 4) -> str:
        return "\n".join((" " * indent) + line if line.strip() else "" for line in code.splitlines())

    
    @with_performance_tracking("async_code_execution")
    @rate_limited("modal")
    async def run_code_async(self, code_or_obj) -> str:
        """
        Execute Python code or a code object in a Modal sandbox asynchronously.
        This method supports both string code and compiled code objects, ensuring
        that the code is executed in a secure, isolated environment with safety checks.
        Args:
            code_or_obj (str or types.CodeType): The Python code to execute, either as a string
                                                 or a compiled code object
        Returns:
            str: The output of the executed code, including any print statements
        """
        await self._ensure_pool_initialized()
        
        if isinstance(code_or_obj, str):
            payload = code_or_obj
        elif isinstance(code_or_obj, types.CodeType):
            b64 = base64.b64encode(marshal.dumps(code_or_obj)).decode()
            payload = textwrap.dedent(f"""
                import base64, marshal, types, traceback
                code = marshal.loads(base64.b64decode({b64!r}))
                try:
                    exec(code, {{'__name__': '__main__'}})
                except Exception:
                    traceback.print_exc()
            """).lstrip()
        else:
            raise CodeExecutionError("Input must be str or types.CodeType")

        # Analyze code for required packages
        start_analysis = time.time()
        required_packages = self._analyze_code_dependencies(payload)
        analysis_time = time.time() - start_analysis
        if analysis_time > 0.1:  # Only log if analysis takes significant time
            logger.info(f"Code dependency analysis took {analysis_time:.2f}s")

        # Add safety shim
        safe_code = self._add_safety_shim(payload)
        filename = "temp_user_code.py"
        write_cmd = f"cat > {filename} <<'EOF'\n{safe_code}\nEOF"

        try:
            async with self.sandbox_pool.get_sandbox() as sb:
                try:
                    # Install additional packages if needed
                    if required_packages:
                        install_start = time.time()
                        await self._install_packages_in_sandbox(sb, required_packages)
                        install_time = time.time() - install_start
                        logger.info(f"Package installation took {install_time:.2f}s")

                    logger.info(f"Writing code to sandbox file: {filename}")
                    sb.exec("bash", "-c", write_cmd)
                    logger.info(f"Executing code from file: {filename}")
                    exec_start = time.time()
                    proc = sb.exec("python", filename)
                    exec_time = time.time() - exec_start
                    logger.info(f"Code execution took {exec_time:.2f}s")
                    
                    output = ""
                    if hasattr(proc, "stdout") and hasattr(proc.stdout, "read"):
                        output = proc.stdout.read()
                        if hasattr(proc, "stderr") and hasattr(proc.stderr, "read"):
                            output += proc.stderr.read()
                    else:
                        output = str(proc)
                    logger.info("Async code execution completed successfully (warm pool)")
                    return output
                except Exception as e:
                    if "finished" in str(e) or "NOT_FOUND" in str(e):
                        logger.warning(f"Sandbox died during use, terminating: {e}")
                        try:
                            result = sb.terminate()
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception as term_e:
                            logger.warning(f"Failed to terminate sandbox after error: {term_e}")
                        async with self.sandbox_pool.get_sandbox() as new_sb:
                            # Re-install packages if needed for retry
                            if required_packages:
                                await self._install_packages_in_sandbox(new_sb, required_packages)
                            new_sb.exec("bash", "-c", write_cmd)
                            proc = new_sb.exec("python", filename)
                            output = ""
                            if hasattr(proc, "stdout") and hasattr(proc.stdout, "read"):
                                output = proc.stdout.read()
                                if hasattr(proc, "stderr") and hasattr(proc.stderr, "read"):
                                    output += proc.stderr.read()
                            else:
                                output = str(proc)
                        logger.info("Async code execution completed successfully on retry")
                        return output
                    else:
                        logger.error(f"Async code execution failed: {e}")
                        raise CodeExecutionError(f"Error executing code in Modal sandbox: {str(e)}")
        except CodeExecutionError:
            raise
        except asyncio.TimeoutError:
            logger.error("Async code execution timed out")
            raise CodeExecutionError("Code execution timed out after 30 seconds")
        except Exception as e:
            logger.error(f"Async code execution failed: {str(e)}")
            raise CodeExecutionError(f"Error executing code in Modal sandbox: {str(e)}")

    def _analyze_code_dependencies(self, code: str) -> List[str]:
        """Analyze code to determine what packages need to be installed."""
        try:
            from mcp_hub.package_utils import extract_imports_from_code, get_packages_to_install
            
            # Extract imports from the code
            detected_imports = extract_imports_from_code(code)
            logger.debug(f"Detected imports: {detected_imports}")
            
            # Determine what packages need to be installed
            packages_to_install = get_packages_to_install(detected_imports)
            
            if packages_to_install:
                logger.info(f"Additional packages needed: {packages_to_install}")
            else:
                logger.debug("No additional packages needed")
                
            return packages_to_install
            
        except Exception as e:
            logger.warning(f"Failed to analyze code dependencies: {e}")
            return []

    async def _install_packages_in_sandbox(self, sandbox: modal.Sandbox, packages: List[str]):
        """Install additional packages in the sandbox."""
        try:
            from mcp_hub.package_utils import create_package_install_command
            
            install_cmd = create_package_install_command(packages)
            if not install_cmd:
                return
                
            logger.info(f"Installing packages: {' '.join(packages)}")
            
            # Execute pip install command
            proc = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: sandbox.exec("bash", "-c", install_cmd, timeout=60)
            )
            
            # Check installation success
            if hasattr(proc, 'stdout') and hasattr(proc.stdout, 'read'):
                output = proc.stdout.read()
                if "Successfully installed" in output or "Requirement already satisfied" in output:
                    logger.info("Package installation completed successfully")
                else:
                    logger.warning(f"Package installation output: {output}")
            
        except Exception as e:
            logger.error(f"Failed to install packages {packages}: {e}")
            # Don't raise exception - continue with execution, packages might already be available

      
    @with_performance_tracking("sync_code_execution")
    @rate_limited("modal")
    def run_code(self, code_or_obj) -> str:
        """
        Execute Python code or a code object in a Modal sandbox synchronously.
        This method supports both string code and compiled code objects, ensuring
        that the code is executed in a secure, isolated environment with safety checks.
        Args:
            code_or_obj (str or types.CodeType): The Python code to execute, either as a string
                                                 or a compiled code object
        Returns:
            str: The output of the executed code, including any print statements
        """
        try:
            logger.info("Executing code synchronously in Modal sandbox")
            
            if isinstance(code_or_obj, str):
                payload = code_or_obj
            elif isinstance(code_or_obj, types.CodeType):
                b64 = base64.b64encode(marshal.dumps(code_or_obj)).decode()
                payload = textwrap.dedent(f"""
                    import base64, marshal, types, traceback
                    code = marshal.loads(base64.b64decode({b64!r}))
                    try:
                        exec(code, {{'__name__': '__main__'}})
                    except Exception:
                        traceback.print_exc()
                """).lstrip()
            else:
                raise CodeExecutionError("Input must be str or types.CodeType")
           
            # Add safety shim
            safe_code = self._add_safety_shim(payload)
            filename = "temp_user_code.py"
            write_cmd = f"cat > {filename} <<'EOF'\n{safe_code}\nEOF"
            
            # Create sandbox synchronously
            sb = None
            try:
                sb = modal.Sandbox.create(
                    app=self.app,
                    image=self.image,
                    cpu=2.0,
                    memory=1024,
                    timeout=35,
                )
                
                sb.exec("bash", "-c", write_cmd)
                proc = sb.exec("python", filename)
                output = ""

                if hasattr(proc, "stdout") and hasattr(proc.stdout, "read"):
                    output = proc.stdout.read()
                    if hasattr(proc, "stderr") and hasattr(proc.stderr, "read"):
                        output += proc.stderr.read()
                else:
                    output = str(proc)
                    
                logger.info("Sync code execution completed successfully")
                return output
                        

            except Exception as e:
                logger.warning(f"Error reading sandbox output: {e}")
                output = str(proc)

            logger.info("Sync code execution completed successfully")
            return output

        except CodeExecutionError:
            raise
        except Exception as e:
            logger.error(f"Sync code execution failed: {str(e)}")
            raise CodeExecutionError(f"Error executing code in Modal sandbox: {str(e)}")
    
    async def cleanup_pool(self):
        """Cleanup the sandbox pool when shutting down."""
        if self.sandbox_pool and self._pool_initialized:
            await self.sandbox_pool.stop()
            logger.info("Sandbox pool cleaned up")

class OrchestratorAgent:
    """
    Main orchestrator that coordinates all agents for the complete workflow.

    This agent manages the end-to-end workflow by coordinating question enhancement,
    web search, LLM processing, citation formatting, code generation, and code execution.
    It provides the primary interface for complex multi-step AI-assisted tasks.
    """
    
    def __init__(self):
        self.question_enhancer = QuestionEnhancerAgent()
        self.web_search = WebSearchAgent()
        self.llm_processor = LLMProcessorAgent()
        self.citation_formatter = CitationFormatterAgent()
        self.code_generator = CodeGeneratorAgent()
        self.code_runner = CodeRunnerAgent()
    
    def orchestrate(self, user_request: str) -> tuple[Dict[str, Any], str]:
        """
        Orchestrate the complete workflow: enhance question → search → generate code → execute.

        Manages the full AI-assisted workflow by coordinating all agents to provide
        comprehensive research, code generation, and execution. Returns both structured
        data and natural language summaries of the complete process.

        Args:
            user_request (str): The user's original request or question

        Returns:
            tuple[Dict[str, Any], str]: A tuple containing the complete result dictionary
                                       and a natural language summary of the process
        """
        try:
            logger.info(f"Starting orchestration for: {user_request[:100]}...")
            
            # Step 1: Enhance the question
            logger.info("Step 1: Enhancing question...")
            enhanced_result = self.question_enhancer.enhance_question(user_request, num_questions=3)
            sub_questions = enhanced_result.get('sub_questions', [user_request])
              # Step 2: Search for information
            logger.info("Step 2: Searching for information...")
            search_results = []            
            search_summaries = []
            
            for i, question in enumerate(sub_questions[:2]):  # Limit to 2 questions to avoid too many searches
                logger.info(f"Processing question {i+1}: {question}")
                try:
                    search_result = self.web_search.search(question)
                    logger.info(f"Search result for question {i+1}: {search_result}")

                    # Extract results and summary regardless of status key
                    results = search_result.get('results', [])
                    summary = search_result.get('tavily_answer', search_result.get('summary', ''))

                    if results or summary:  # Treat as success if any results or summary found
                        logger.info(f"Question {i+1} - Found {len(results)} results")
                        logger.info(f"Question {i+1} - Summary: {summary[:100]}...")

                        # Add to collections
                        search_results.extend(results)
                        search_summaries.append(summary)

                        logger.info(f"Question {i+1} - Successfully added {len(results)} results to collection")
                        logger.info(f"Question {i+1} - Current total search_results: {len(search_results)}")
                        logger.info(f"Question {i+1} - Current total search_summaries: {len(search_summaries)}")
                    else:
                        error_msg = search_result.get('error', 'Unknown error or no results returned') 
                        logger.warning(f"Search failed for question {i+1}: {error_msg}")

                except Exception as e:
                    logger.error(f"Exception during search for question '{question}': {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
            
            logger.info(f"Total search results collected: {len(search_results)}")
            logger.info(f"Total search summaries: {len(search_summaries)}")
            for i, result in enumerate(search_results[:3]):
                logger.info(f"Search result {i+1}: {result.get('title', 'No title')[:50]}...")
            
            # Step 3: Create grounded context
            logger.info("Step 3: Creating grounded context...")
            grounded_context = ""
            if search_results:
                # Combine search results into context
                context_parts = []
                for result in search_results[:5]:  # Limit to top 5 results
                    context_parts.append(f"Title: {result.get('title', 'N/A')}")
                    context_parts.append(f"Content: {result.get('content', 'N/A')}")
                    context_parts.append(f"URL: {result.get('url', 'N/A')}")
                    context_parts.append("---")
                
                grounded_context = "\n".join(context_parts)
            
            # If no search results, use a generic context
            if not grounded_context:
                grounded_context = f"User request: {user_request}\nNo additional web search context available."
              # Step 4: Generate code
            logger.info("Step 4: Generating code...")
            logger.info(f"Grounded context length: {len(grounded_context)}")
            code_result, code_summary = self.code_generator.generate_code(user_request, grounded_context)
            logger.info(f"Code generation result: {code_result}")
            logger.info(f"Code generation summary: {code_summary[:200]}...")
            
            code_string = ""
            if code_result.get('status') == 'success':
                # Use raw_output (string) for display, generated_code (compiled) for execution
                code_string = code_summary  # This is the raw string output
                logger.info(f"Successfully extracted code_string with length: {len(code_string)}")
                logger.info(f"Code preview: {code_string[:200]}...")
            else:
                logger.warning(f"Code generation failed: {code_result.get('error', 'Unknown error')}")
            
            # Step 5: Execute code if available
            execution_output = ""
            if code_string:
                logger.info("Step 5: Executing code...")
                try:
                    # Use async execution for better performance
                    import asyncio
                    execution_output = asyncio.run(self.code_runner.run_code_async(code_string))
                except Exception as e:
                    execution_output = f"Execution failed: {str(e)}"
                    logger.warning(f"Code execution failed: {e}")
            
            # Step 6: Format citations
            logger.info("Step 6: Formatting citations...")
            citations = []
            for result in search_results:
                if result.get('url'):
                    citations.append(f"{result.get('title', 'Untitled')} - {result.get('url')}")
              # Compile final result
            logger.info("=== PRE-FINAL RESULT DEBUG ===")
            logger.info(f"search_results length: {len(search_results)}")
            logger.info(f"search_summaries length: {len(search_summaries)}")
            logger.info(f"code_string length: {len(code_string)}")
            logger.info(f"execution_output length: {len(execution_output)}")
            logger.info(f"citations length: {len(citations)}")
            

            logger.info("=== GENERATING EXECUTIVE SUMMARY ===")
            # Sample first search result
            if search_results:
                logger.info(f"First search result: {search_results[0]}")

            prompt = f"""
            The user asked about {user_request} which yielded this summary: {search_summaries} 
            
            During the orchestration, you generated the following code: {code_string}

            The code was executed in a secure sandbox environment, and the output was <executed_code>{execution_output}</executed_code>.

            If there was no output in the executed_code tags, please state how to answer the user's request showing the code required.
            State that the code you are giving them has not been executed, and that they should run it in their own environment.

            Please provide a short and concise summary of the code that you wrote, including the user request, the summaries provided and the code generated.
            Explain how the code addresses the user's request, what it does, and any important details about its execution.

            Touch upon the other methods available that were found in the search results, and how they relate to the user's request.
            
            Please return the result in natural language only, without any code blocks, unless as stated above, there was no code executed in the sandbox and then you should give them the code
            as a code block.
            References to code can be made to explain why particular code has been used regardless of sandbox execution, e.g. discuss why the LinerRegression module was used  from scikit-learn etc.
            
            If no code was generated, apologise, please state that clearly the code generation failed in the sandbox, this could be due to restriction
            or the code being too complex for the sandbox to handle.

            Note, if appropriate, indicate how the code can be modified to include human input etc. as this is a banned keyword in the sandbox.

            The response should be directed at the user, in a friendly and helpful manner, as if you were a human assistant helping the user with their request.

            **Summary Requirements:**

            - The summary should be concise, no more than 500 words.
            - It should clearly explain how the code addresses the user's request.
            - It should only include code if there was no execution output, and then it should be in a code block. (if there is executed_code, this will be returned by
            another process and therefor you dont need to do it here)
            - The summary should be written in a friendly and helpful tone, as if you were a human assistant helping the user with their request.

            """

            messages = [{"role": "user", 
                         "content": prompt}]
            
            logger.info(f"LLM provider is: {api_config.llm_provider}, model used: {model_config.get_model_for_provider('llm_processor', api_config.llm_provider)}")
            # Last call to LLM to summarize the entire orchestration
            overall_summary = make_llm_completion(
                model=model_config.get_model_for_provider("llm_processor", api_config.llm_provider),
                messages=messages,
                temperature=app_config.llm_temperature
            )            
            logger.info("Overall summary generated:")
            
            final_result = {
                "status": "success",
                "user_request": user_request,
                "sub_questions": sub_questions,
                "search_results": search_results[:5],
                "search_summaries": search_summaries,
                "code_string": code_string,
                "execution_output": execution_output,
                "citations": citations,
                "final_summary": f"{overall_summary}",
                "message": "Orchestration completed successfully"
            }
            
            # Create clean summary for display
            final_narrative = f"## 🎯 Request: {user_request}\n\n{overall_summary}"
            
            logger.info("Orchestration completed successfully")
            return final_result, final_narrative
            
        except (ValidationError, APIError, CodeGenerationError) as e:
            logger.error(f"Orchestration failed: {str(e)}")
            # Create execution log for error case
            execution_log = f"Error during orchestration: {str(e)}"
            return {"error": str(e), "execution_log": execution_log}, str(e)
        except Exception as e:
            logger.error(f"Unexpected error in orchestration: {str(e)}")
            # Create execution log for error case
            execution_log = f"Unexpected error: {str(e)}"
            return {"error": f"Unexpected error: {str(e)}", "execution_log": execution_log}, str(e)
    
    def _format_search_results(self, results):
        """Format search results into a combined text snippet."""
        formatted_parts = []
        for result in results:
            title = result.get('title', 'No title')
            content = result.get('content', 'No content')
            url = result.get('url', 'No URL')
            formatted_parts.append(f"Title: {title}\nContent: {content}\nURL: {url}\n---")
        
        return "\n".join(formatted_parts)
    
    async def _run_subquestion_async(self, sub_question: str, user_request: str) -> tuple:
        """Process a single sub-question asynchronously."""
        try:
            # Search
            search_result = await self.web_search.search_async(sub_question)
            if search_result.get("error"):
                logger.warning(f"Async search failed for sub-question: {search_result['error']}")
                return None, None
            
            # Format search results
            results = search_result.get("results", [])[:app_config.max_search_results]
            formatted_text = self._format_search_results(results)
            
            # Process search results
            llm_summary = await self.llm_processor.async_process(
                formatted_text, 
                "summarize", 
                f"Context of user request: {user_request}"
            )
            
            # Prepare result
            result_data = {
                "status": "success",
                "sub_question": sub_question,
                "user_request": user_request,
                "search_results": results,
                "search_summary": llm_summary.get('llm_processed_output', '')
            }
            
            # Create summary parts
            summary_parts = []
            summary_parts.append(f"## Subquestion: {sub_question}")
            summary_parts.append("### Research Summary:")
            summary_parts.append(llm_summary.get('llm_processed_output', 'No summary available'))
            
            # Add sources if available
            citations = []
            for result in results:
                if result.get('url'):
                    citations.append(f"{result.get('title', 'Untitled')} - {result.get('url')}")
            
            if citations:
                summary_parts.append("### Sources:")
                for i, citation in enumerate(citations, 1):
                    summary_parts.append(f"{i}. {citation}")
            
            clean_summary = "\n\n".join(summary_parts)
            
            logger.info("Subquestion processing completed successfully")
            return result_data, clean_summary
            
        except Exception as e:
            logger.error(f"Subquestion processing failed: {e}")
            error_result = {
                "status": "error",
                "user_request": user_request,
                "sub_question": sub_question,
                "error": str(e),
                "message": "Subquestion processing failed"
            }
            return error_result, f"❌ Error: {str(e)}"

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

