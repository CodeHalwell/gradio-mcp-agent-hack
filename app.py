"""Enhanced MCP Hub - Single Unified Version with Advanced Features."""
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
from typing import Dict, Any, Optional
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
    """Decorator to add performance tracking to any function (sync or async)."""
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
    """Agent responsible for enhancing questions into sub-questions."""
    
    @with_performance_tracking("question_enhancement")
    @rate_limited("nebius")
    @circuit_protected("nebius")
    @cached(ttl=300)  # Cache for 5 minutes
    def enhance_question(self, user_request: str, num_questions: int) -> Dict[str, Any]:
        """Split a single user query into three distinct sub-questions."""
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
    """Agent responsible for performing web searches."""
    
    def __init__(self):
        if not api_config.tavily_api_key:
            raise APIError("Tavily", "API key not configured")
        self.client = TavilyClient(api_key=api_config.tavily_api_key)
    
    @with_performance_tracking("web_search")
    @rate_limited("tavily")
    @circuit_protected("tavily")
    @cached(ttl=600)  # Cache for 10 minutes
    def search(self, query: str) -> Dict[str, Any]:
        """Perform a web search using the Tavily API."""
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
        """Perform an async web search using aiohttp."""
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
    """Agent responsible for LLM processing tasks."""
    
    @with_performance_tracking("llm_processing")
    @rate_limited("nebius")
    @circuit_protected("nebius")
    def process(self, text_input: str, task: str, context: str = None) -> Dict[str, Any]:
        """Process text using LLM for summarization, reasoning, or keyword extraction."""
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
        """Process text using async LLM for summarization, reasoning, or keyword extraction."""
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
            "reason": f"Analyze this text and provide detailed reasoning:\n\n{text_input}",
            "summarize": f"Summarize in detail while being concise:\n\n{text_input}",
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
    """Agent responsible for formatting citations."""
    
    @with_performance_tracking("citation_formatting")
    def format_citations(self, text_block: str) -> Dict[str, Any]:
        """Extract URLs and produce APA-style citations."""
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
    """Agent responsible for generating Python code."""

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
        return f"""
                You are an expert Python developer. **Rules**:
                - Never use these functions: {disallowed_list}
                - Never import os, subprocess, or sys modules
                - After defining functions/classes, call them and print the result.
                - Always include print statements to show output
                {f"Previous attempt failed:\n{prev_err}\nFix it." if prev_err else ""}

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

                NEVER include input()

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

                """

    @with_performance_tracking("code_generation")
    @rate_limited("nebius")
    @circuit_protected("nebius")
    def generate_code(
        self, user_request: str, grounded_context: str
    ) -> tuple[Dict[str, Any], str]:
        """Generate Python code based on user request and grounded context with enhanced security."""
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
                    )                    # Log the generated code first for debugging
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

                    return {"generated_code": code_compiled}, raw_output

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
    """Agent responsible for running code in Modal sandbox with enhanced security."""
    
    def __init__(self):
        self.app = modal.App.lookup(app_config.modal_app_name, create_if_missing=True)
        # Create enhanced image with common packages for better performance
        self.image = self._create_enhanced_image()
    
    def _create_enhanced_image(self):
        """Create a Modal image with commonly used packages pre-installed."""
        common_packages = [
            "numpy",
            "pandas",
            "polars",
            "matplotlib",
            "seaborn",
            "plotly",
            "scikit-learn",
            "lightgbm",
            "xgboost",
            "requests",
            "beautifulsoup4",
            "scrapy",
            "flask",
            "fastapi",
            "starlette",
            "pillow",
            "imageio",
            "tqdm",
            "pytest",
            "python-dateutil",
            "pydantic",
            "click",
            "rich",
            "httpx",
            "duckdb",
            "networkx",
            "schedule",
            "watchdog",
            "sqlalchemy",
        ]
        
        try:
            return modal.Image.debian_slim().pip_install(*common_packages)
        except Exception as e:
            logger.warning(f"Failed to create enhanced image, using basic: {e}")
            return modal.Image.debian_slim()
    
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
        """Add safety shim to wrap user code with security constraints."""
        try:          
            # Add a safety wrapper around the user code
            safety_shim = f'''
import sys
import types           # needed for CodeType and MappingProxyType
import functools
import builtins
import marshal
import traceback

# Built-ins the user code must NOT touch
RESTRICTED_BUILTINS = {{
    'open', 'input', 'eval', 'compile', '__import__',
    'getattr', 'setattr', 'delattr', 'hasattr', 'globals', 'locals',
    'pty', 'subprocess', 'socket', 'threading', 'ssl', 'email', 'smtpd'
    
}}

# Snapshot current built-ins (works whether __builtins__ is a dict or a module)
if isinstance(__builtins__, dict):
    _original_builtins = __builtins__.copy()
else:
    _original_builtins = __builtins__.__dict__.copy()

# Remove restricted names
_safe_builtins = {{k: v for k, v in _original_builtins.items()
                   if k not in RESTRICTED_BUILTINS}}

# Allow print for user feedback
_safe_builtins['print'] = print

# Guarded exec that only accepts pre-compiled CodeType objects
def safe_exec(code_obj, globals_dict=None, locals_dict=None):
    if not isinstance(code_obj, types.CodeType):
        raise TypeError("safe_exec only accepts a compiled code object")
    if globals_dict is None:
        globals_dict = {{"__builtins__": types.MappingProxyType(_safe_builtins)}}
    return _original_builtins['exec'](code_obj, globals_dict, locals_dict)

_safe_builtins['exec'] = safe_exec  # expose to user code

# Import hook: allow only the modules we list explicitly
def safe_import(name, *args, **kwargs):
    ALLOWED_MODULES = (
    set(sys.stdlib_module_names)          # every standard-lib top-level pkg
    .difference(RESTRICTED_BUILTINS)                     # …minus the risky ones
    .union(                               # …plus the external stack you need
        {{
            # scientific / viz stack
            "numpy", "pandas", "polars",
            "matplotlib", "seaborn", "plotly",

            # machine learning / modelling
            "scikit-learn", "lightgbm", "xgboost",

            # data / web access & scraping
            "requests", "httpx", "beautifulsoup4", "scrapy",

            # web frameworks / ASGI / WSGI
            "flask", "fastapi", "starlette",

            # imaging & media
            "pillow", "imageio",

            # progress / CLI / pretty-printing
            "tqdm", "click", "rich",

            # scheduling & filesystem watching
            "schedule", "watchdog",

            # analytics / storage
            "duckdb", "sqlalchemy",

            # graphs / networks
            "networkx",

            # data-validation / models
            "pydantic",

            # date & time utils
            "python-dateutil",

            # testing
            "pytest",
        }}
    )
)
    if name in ALLOWED_MODULES:
        return _original_builtins['__import__'](name, *args, **kwargs)
    raise ImportError(f"Module {{name!r}} is not allowed in this environment")

_safe_builtins['__import__'] = safe_import

# Run the user’s code inside the restricted environment
try:
    exec(
        """{code}""",
        {{"__builtins__": types.MappingProxyType(_safe_builtins)}}
    )
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    traceback.print_exc()
'''
            return safety_shim
            
        except Exception as e:
            logger.error(f"Failed to add safety shim: {str(e)}")
            raise CodeExecutionError(f"Failed to prepare safe code execution: {str(e)}")
    
    @with_performance_tracking("async_code_execution")
    @rate_limited("modal")
    async def run_code_async(self, code_or_obj) -> str:
        """Execute code asynchronously in Modal sandbox with enhanced safety."""
        try:
            logger.info("Executing code asynchronously in enhanced Modal sandbox")
            
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
            
            # Add enhanced safety shim
            payload = self._add_safety_shim(payload)
            
            # Create sandbox with async context manager for better resource management
            async with self._create_async_sandbox() as sb:
                # Use asyncio to handle the execution with timeout
                proc = await asyncio.wait_for(
                    self._execute_in_sandbox_async(sb, payload),
                    timeout=30
                )
                output = proc.stdout.read() + proc.stderr.read()
                logger.info("Async code execution completed successfully")
                return output
                        
        except CodeExecutionError:
            raise
        except asyncio.TimeoutError:
            logger.error("Async code execution timed out")
            raise CodeExecutionError("Code execution timed out after 30 seconds")
        except Exception as e:
            logger.error(f"Async code execution failed: {str(e)}")
            raise CodeExecutionError(f"Error executing code in Modal sandbox: {str(e)}")
    
    @asynccontextmanager
    async def _create_async_sandbox(self):
        """Create and manage Modal sandbox asynchronously."""
        sb = None
        try:
            # Create sandbox with enhanced configuration
            sb = modal.Sandbox.create(
                app=self.app,
                image=self.image,
                cpu=2.0,  # Increased CPU for better performance
                memory=1024,  # Increased memory
                timeout=35,
                environment={
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONDONTWRITEBYTECODE": "1"
                }
            )
            yield sb
        finally:
            if sb:
                try:
                    await asyncio.get_event_loop().run_in_executor(None, sb.terminate)
                except Exception as e:
                    logger.warning(f"Failed to terminate async sandbox: {str(e)}")
    
    async def _execute_in_sandbox_async(self, sb, payload):
        """Execute payload in sandbox asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, sb.exec, "python", "-c", payload)

    @with_performance_tracking("sync_code_execution")
    @rate_limited("modal")
    def run_code(self, code_or_obj) -> str:
        """Execute code synchronously in Modal sandbox with enhanced safety."""
        try:
            logger.info("Executing code synchronously in enhanced Modal sandbox")
            
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
            
            # Add enhanced safety shim
            payload = self._add_safety_shim(payload)
            
            # Create sandbox synchronously
            try:
                sb = modal.Sandbox.create(
                    app=self.app,
                    image=self.image,
                    cpu=2.0,
                    memory=1024,
                    timeout=35,
                )
                
                # Execute with timeout
                proc = sb.exec("python", "-c", payload, timeout=30)
                output = proc.stdout.read() + proc.stderr.read()
                if 'error' in output.lower():
                    logger.error(f"Code execution returned an error: {output}")
                    raise CodeExecutionError(f"Code execution failed with error: {output}")
                    return output
                logger.info("Sync code execution completed successfully")
                return output
                        
            except Exception as e:
                logger.error(f"Sync code execution failed: {str(e)}")
                raise CodeExecutionError(f"Error executing code in Modal sandbox: {str(e)}")
            finally:
                try:
                    if sb:
                        sb.terminate()
                except Exception as e:
                    logger.warning(f"Failed to terminate sync sandbox: {str(e)}")
                        
        except CodeExecutionError:
            raise
        except Exception as e:
            logger.error(f"Sync code execution failed: {str(e)}")
            raise CodeExecutionError(f"Error executing code in Modal sandbox: {str(e)}")


class OrchestratorAgent:
    """Main orchestrator that coordinates all agents for the complete workflow."""
    
    def __init__(self):
        self.question_enhancer = QuestionEnhancerAgent()
        self.web_search = WebSearchAgent()
        self.llm_processor = LLMProcessorAgent()
        self.citation_formatter = CitationFormatterAgent()
        self.code_generator = CodeGeneratorAgent()
        self.code_runner = CodeRunnerAgent()
    
    def orchestrate(self, user_request: str) -> tuple[Dict[str, Any], str]:
        """
        Orchestrate the complete workflow:
        1) Enhance question → 3 sub-questions
        2) For each sub-question: search → summarize → cite
        3) Combine summaries into grounded context
        4) Generate Python code
        5) Execute code
        6) Return results with natural language summary
        """
        execution_log = []
        
        try:
            validate_non_empty_string(user_request, "User request")
            logger.info(f"Starting orchestration for request: {user_request[:100]}...")
            
            # Step 1: Enhance into sub-questions
            logger.info("Step 1: Enhancing question into sub-questions")
            enhancer_result = self.question_enhancer.enhance_question(user_request, num_questions=2)
            execution_log.append({
                "step": 1,
                "tool": "question_enhancer",
                "input": user_request,
                "result": enhancer_result
            })
            
            if enhancer_result.get("error"):
                raise ValidationError(f"Question enhancement failed: {enhancer_result['error']}")
            
            sub_questions = enhancer_result.get("sub_questions", [])
            if not sub_questions:
                raise ValidationError("No sub-questions returned.")
            
            # Step 2: Process each sub-question
            logger.info(f"Step 2: Processing {len(sub_questions)} sub-questions")
            all_sub_summaries = []
            all_citations = []
            citation_errors = []
            
            for idx, sub_q in enumerate(sub_questions, start=1):
                logger.info(f"Processing sub-question {idx}: {sub_q}")
                
                # 2a. Web search
                search_result = self.web_search.search(sub_q)
                execution_log.append({
                    "step": f"2a_subquestion_{idx}_search",
                    "tool": "web_search",
                    "query": sub_q,
                    "result": search_result
                })
                
                if search_result.get("error"):
                    logger.warning(f"Search failed for sub-question {idx}: {search_result['error']}")
                    continue
                
                # Build combined snippet from results
                results = search_result.get("results", [])[:app_config.max_search_results]
                combined_snippet = self._format_search_results(results)
                
                # 2b. Summarization
                summary_result = self.llm_processor.process(
                    combined_snippet, 
                    "summarize", 
                    context=None
                )
                execution_log.append({
                    "step": f"2b_subquestion_{idx}_summarize",
                    "tool": "llm_processor",
                    "input": combined_snippet,
                    "result": summary_result
                })
                
                if summary_result.get("error"):
                    logger.warning(f"Summarization failed for sub-question {idx}: {summary_result['error']}")
                    continue
                
                sub_summary = summary_result.get("llm_processed_output", "")
                all_sub_summaries.append(sub_summary)
                
                # 2c. Citation formatting
                citation_result = self.citation_formatter.format_citations(combined_snippet)
                execution_log.append({
                    "step": f"2c_subquestion_{idx}_citations",
                    "tool": "citation_formatter",
                    "input": combined_snippet,
                    "result": citation_result
                })
                
                if citation_result.get("error"):
                    citation_errors.append(f"Sub-question {idx}: {citation_result['error']}")
                else:
                    all_citations.extend(citation_result.get("formatted_citations", []))
            
            # Step 3: Combine summaries
            logger.info("Step 3: Combining sub-summaries")
            if not all_sub_summaries:
                raise ValidationError("No successful summaries generated from sub-questions.")
            
            combine_input = "\n\n".join([
                f"Summary {i+1}:\n{sub}" for i, sub in enumerate(all_sub_summaries)
            ])
            
            combine_result = self.llm_processor.process(
                combine_input,
                "summarize",
                context="Combine these sub-summaries into a single cohesive context for code generation."
            )
            execution_log.append({
                "step": 3,
                "tool": "llm_processor_combine",
                "input": combine_input,
                "result": combine_result
            })
            
            if combine_result.get("error"):
                raise ValidationError(f"Failed to combine summaries: {combine_result['error']}")
            
            final_summary = combine_result.get("llm_processed_output", "")
            
            # Step 4: Generate code
            logger.info("Step 4: Generating Python code")
            grounded_context = final_summary + "\n\nCitations:\n" + "\n".join(all_citations)
            
            code_result, code_string = self.code_generator.generate_code(user_request, grounded_context)
            execution_log.append({
                "step": 4,
                "tool": "code_generator",
                "input": {"user_request": user_request, "grounded_context": grounded_context},
                "result": code_result
            })
            
            if code_result.get("error"):
                raise CodeGenerationError(f"Code generation failed: {code_result['error']}")
            
            generated_code = code_result.get("generated_code", "")
              # Step 5: Execute code
            logger.info("Step 5: Executing generated code")
            try:
                code_output = self.code_runner.run_code(generated_code)
            except CodeExecutionError as e:
                code_output = f"Code execution failed: {str(e)}"
                logger.warning(f"Code execution failed: {str(e)}")
            
            execution_log.append({
                "step": 5,
                "tool": "code_runner",
                "input": generated_code,
                "result": {"code_string": code_string, "code_output": code_output}
            })
            
            # Step 6: Generate final natural language summary
            logger.info("Step 6: Generating final summary")
            summary_prompt = f"""
            Summarize the entire research and code generation process:
            User Request: {user_request}
            Research Summary: {final_summary}
            Generated Code: {code_string}
            Code Output: {code_output}
            Citations: {', '.join(all_citations)}
            
            Provide a concise summary of the entire process, including the user request, 
            research findings, generated code, execution output, and citations.
            """
            
            try:
                logger.info("Generating final narrative using LLM")
                logger.info(f"LLM provider is: {api_config.llm_provider}, model used: {model_config.get_model_for_provider('orchestrator', api_config.llm_provider)}")
                final_narrative = make_llm_completion(
                    model=model_config.get_model_for_provider("orchestrator", api_config.llm_provider),
                    messages=[{"role": "user", "content": summary_prompt}],
                    temperature=0.5
                )
            except APIError as e:
                logger.warning(f"Failed to generate final narrative: {str(e)}")
                final_narrative = "Process completed successfully. Generated and executed code based on research findings."
            
            citation_error = "; ".join(citation_errors) if citation_errors else None
            
            result = {
                "user_request": user_request,
                "final_summary": final_summary,
                "code_string": code_string,
                "generated_code": generated_code,
                "code_output": code_output,
                "citations": all_citations,
                "citation_error": citation_error,
                "execution_log": execution_log
            }
            
            logger.info("Orchestration completed successfully")
            return result, final_narrative
            
        except (ValidationError, APIError, CodeGenerationError) as e:
            logger.error(f"Orchestration failed: {str(e)}")
            return {"error": str(e), "execution_log": execution_log}, str(e)
        except Exception as e:
            logger.error(f"Unexpected error in orchestration: {str(e)}")
            return {"error": f"Unexpected error: {str(e)}", "execution_log": execution_log}, str(e)
    
    async def _run_subquestion_async(self, sub_question: str) -> tuple:
        """Process a single sub-question asynchronously."""
        try:
            # Search
            search_result = await self.web_search.async_search(sub_question)
            if search_result.get("error"):
                logger.warning(f"Async search failed for sub-question: {search_result['error']}")
                return None, None
            
            # Format search results
            results = search_result.get("results", [])[:app_config.max_search_results]
            combined_snippet = self._format_search_results(results)
            
            # Summarize
            summary_result = await self.llm_processor.async_process(
                combined_snippet, 
                "summarize", 
                context=None
            )
            
            if summary_result.get("error"):
                logger.warning(f"Async summarization failed for sub-question: {summary_result['error']}")
                return None, None
            
            sub_summary = summary_result.get("llm_processed_output", "")
            
            # Format citations
            citation_result = self.citation_formatter.format_citations(combined_snippet)
            citations = citation_result.get("formatted_citations", [])
            
            return sub_summary, citations
            
        except Exception as e:
            logger.error(f"Error processing sub-question async: {e}")
            return None, None

    async def orchestrate_async(self, user_request: str) -> tuple[Dict[str, Any], str]:
        """
        Async orchestrate with concurrent sub-question processing for improved performance.
        """
        execution_log = []
        
        try:
            validate_non_empty_string(user_request, "User request")
            logger.info(f"Starting async orchestration for request: {user_request[:100]}...")
            
            # Step 1: Enhance into sub-questions
            logger.info("Step 1: Enhancing question into sub-questions")
            enhancer_result = self.question_enhancer.enhance_question(user_request, num_questions=2)
            execution_log.append({
                "step": 1,
                "tool": "question_enhancer",
                "input": user_request,
                "result": enhancer_result
            })
            
            if enhancer_result.get("error"):
                raise ValidationError(f"Question enhancement failed: {enhancer_result['error']}")
            
            sub_questions = enhancer_result.get("sub_questions", [])
            if not sub_questions:
                raise ValidationError("No sub-questions returned.")
            
            # Step 2: Process sub-questions concurrently with semaphore for rate limiting
            logger.info(f"Step 2: Processing {len(sub_questions)} sub-questions concurrently")
            
            # Limit concurrent requests to avoid overwhelming APIs
            semaphore = asyncio.Semaphore(3)
            
            async def process_with_semaphore(sq):
                async with semaphore:
                    return await self._run_subquestion_async(sq)
            
            # Run all sub-questions concurrently
            tasks = [asyncio.create_task(process_with_semaphore(sq)) for sq in sub_questions]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            all_sub_summaries = []
            all_citations = []
            citation_errors: list = []
            
            for idx, result in enumerate(results, start=1):
                if isinstance(result, Exception):
                    logger.error(f"Sub-question {idx} failed with exception: {result}")
                    continue
                
                sub_summary, citations = result
                if sub_summary:
                    all_sub_summaries.append(sub_summary)
                if citations:
                    all_citations.extend(citations)
            
            if not all_sub_summaries:
                raise ValidationError("No successful summaries generated from sub-questions.")
            
            # Step 3: Combine summaries using batch processing
            logger.info("Step 3: Combining sub-summaries with batch processing")
            
            # Create a composite prompt for batch processing
            batch_prompt = f"""
                    You are an expert researcher. For each summary below, provide a concise technical analysis,
                    then create one overarching summary that combines all insights.

                    Return valid JSON only:
                    {{
                    "individual_analyses": ["...", "...", "..."],
                    "final_summary": "..."
                    }}

                    SUMMARIES:
                    {json.dumps({"summaries": all_sub_summaries})}
            """
            
            from mcp_hub.utils import make_async_llm_completion
            batch_result = await make_async_llm_completion(
                model=model_config.get_model_for_provider("llm_processor", api_config.llm_provider),
                messages=[{"role": "user", "content": batch_prompt}],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            try:
                batch_data = extract_json_from_text(batch_result)
                final_summary = batch_data.get("final_summary", "\n\n".join(all_sub_summaries))
            except Exception as e:
                logger.warning(f"Failed to parse batch result, using fallback: {e}")
                final_summary = "\n\n".join(all_sub_summaries)
            
            execution_log.append({
                "step": 3,
                "tool": "llm_processor_batch",
                "input": all_sub_summaries,
                "result": {"final_summary": final_summary}
            })
            
            # Step 4: Generate code
            logger.info("Step 4: Generating Python code")
            grounded_context = final_summary + "\n\nCitations:\n" + "\n".join(all_citations)
            
            code_result, code_string = self.code_generator.generate_code(user_request, grounded_context)
            execution_log.append({
                "step": 4,
                "tool": "code_generator",
                "input": {"user_request": user_request, "grounded_context": grounded_context},
                "result": code_result
            })
            
            if code_result.get("error"):
                raise CodeGenerationError(f"Code generation failed: {code_result['error']}")
            
            generated_code = code_result.get("generated_code", "")
            
            # Step 5: Execute code (with retry logic as suggested)
            logger.info("Step 5: Executing generated code with retry logic")
            code_output = ""
            max_runtime_attempts = 2
            for attempt in range(1, max_runtime_attempts + 1):
                try:
                    code_output = self.code_runner.run_code(generated_code)
                    break  # Success, exit retry loop
                except CodeExecutionError as e:
                    logger.warning(f"Code execution attempt {attempt} failed: {str(e)}")
                    if attempt < max_runtime_attempts:
                        # Regenerate code with error feedback
                        error_context = f"Previous execution failed: {str(e)}"
                        code_result, code_string = self.code_generator.generate_code(
                            user_request, 
                            grounded_context + f"\n\nError to fix: {error_context}"
                        )
                        if not code_result.get("error"):
                            generated_code = code_result.get("generated_code", "")
                    else:
                        code_output = f"Code execution failed after {max_runtime_attempts} attempts: {str(e)}"
            
            execution_log.append({
                "step": 5,
                "tool": "code_runner",
                "input": generated_code,
                "result": {"code_string": code_string, "code_output": code_output}
            })
            
            # Step 6: Generate final narrative
            logger.info("Step 6: Generating final summary")
            summary_prompt = f"""
            Summarize the entire research and code generation process:
            User Request: {user_request}
            Research Summary: {final_summary}
            Generated Code: {code_string}
            Code Output: {code_output}
            Citations: {', '.join(all_citations)}
            
            Provide a concise summary of the entire process.
            """
            
            try:
                final_narrative = await make_async_llm_completion(
                    model=model_config.get_model_for_provider("orchestrator", api_config.llm_provider),
                    messages=[{"role": "user", "content": summary_prompt}],
                    temperature=0.5
                )
            except APIError as e:
                logger.warning(f"Failed to generate final narrative: {str(e)}")
                final_narrative = "Process completed successfully with async orchestration."
            
            citation_error = "; ".join(citation_errors) if citation_errors else None
            
            result = {
                "user_request": user_request,
                "final_summary": final_summary,
                "code_string": code_string,
                "generated_code": generated_code,
                "code_output": code_output,
                "citations": all_citations,
                "citation_error": citation_error,
                "execution_log": execution_log,
                "async_processing": True,
                "sub_summaries": all_sub_summaries
            }
            
            logger.info("Async orchestration completed successfully")
            return result, final_narrative
            
        except (ValidationError, APIError, CodeGenerationError) as e:
            logger.error(f"Async orchestration failed: {str(e)}")
            return {"error": str(e), "execution_log": execution_log}, str(e)
        except Exception as e:
            logger.error(f"Unexpected error in async orchestration: {str(e)}")
            return {"error": f"Unexpected error: {str(e)}", "execution_log": execution_log}, str(e)

    def _format_search_results(self, results: list) -> str:
        """Format search results into a combined snippet for processing."""
        if not results:
            return "No search results found."
        
        formatted_parts = []
        for i, result in enumerate(results[:3], 1):  # Limit to top 3 results
            title = result.get("title", "No title")
            content = result.get("content", "No content")
            url = result.get("url", "No URL")
            
            # Truncate content if too long
            if len(content) > 500:
                content = content[:500] + "..."
            
            formatted_parts.append(f"""
--- Result {i} ---
Title: {title}
URL: {url}
Content: {content}
""")
        
        return "\n".join(formatted_parts)
    
    def _create_final_summary(self, user_request: str, sub_summaries: list, 
                            code_string: str = None, execution_output: str = None) -> str:
        """Create a comprehensive final summary."""
        summary_parts = [
            f"## Response to: {user_request}\n",
            "### Research Summary",
            "Based on comprehensive research across multiple sources:\n"
        ]
        
        # Add sub-summaries
        for i, summary in enumerate(sub_summaries, 1):
            if summary and summary.strip():
                summary_parts.append(f"**{i}.** {summary}")
        
        # Add code section if available
        if code_string:
            summary_parts.extend([
                "\n### Generated Code",
                f"```python\n{code_string}\n```"
            ])
        
        # Add execution results if available
        if execution_output:
            summary_parts.extend([
                "\n### Execution Results",
                f"```\n{execution_output}\n```"
            ])
        
        return "\n".join(summary_parts)

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
    """Wrapper for OrchestratorAgent."""
    return orchestrator.orchestrate(user_request)

def agent_orchestrator_dual_output(user_request: str) -> tuple:
    """Wrapper for OrchestratorAgent that returns both JSON and natural language output."""
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
    
    return json_result, summary

def agent_question_enhancer(user_request: str) -> dict:
    """Wrapper for QuestionEnhancerAgent."""
    return question_enhancer.enhance_question(user_request, num_questions=2)

def agent_web_search(query: str) -> dict:
    """Wrapper for WebSearchAgent."""
    return web_search.search(query)

def agent_llm_processor(text_input: str, task: str, context: Optional[str] = None) -> dict:
    """Wrapper for LLMProcessorAgent."""
    return llm_processor.process(text_input, task, context)

def agent_citation_formatter(text_block: str) -> dict:
    """Wrapper for CitationFormatterAgent."""
    return citation_formatter.format_citations(text_block)

def agent_code_generator(user_request: str, grounded_context: str) -> tuple:
    """Wrapper for CodeGeneratorAgent."""
    return code_generator.generate_code(user_request, grounded_context)

def code_runner_wrapper(code_or_obj) -> str:
    """Wrapper for CodeRunnerAgent."""
    try:
        return code_runner.run_code(code_or_obj)
    except CodeExecutionError as e:
        return str(e)

# ----------------------------------------
# Advanced Feature Functions
# ----------------------------------------

def get_health_status() -> Dict[str, Any]:
    """Get comprehensive system health status."""
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
    """Get performance metrics and analytics."""
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

class IntelligentCacheManager:
    """Advanced caching system for MCP Hub operations."""
    
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
        """Generate a unique cache key based on operation and parameters."""
        import hashlib
        key_data = f"{operation}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, operation: str, **kwargs):
        """Get cached result if available and not expired."""
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


def process_orchestrator_request(user_request):
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
                        clean_summary += f"### {i}. {summary[:100]}...\n"
            
            if not clean_summary:
                clean_summary = "## ⚠️ Processing Complete\nThe request was processed but no detailed results were generated."
            
            return json_result, clean_summary
# ----------------------------------------
# Gradio UI / MCP Server Setup
# ----------------------------------------

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


with gr.Blocks(title="Shallow Research & Code Assistant Hub", 
               theme=gr.themes.Ocean(),
               fill_width=False,
               css=CUSTOM_CSS) as demo:
    
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                <h1 class="app-title" style="text-align: center; font-size: 2.5rem;">
                    Shallow Research &amp; Code Assistant Hub
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
                <code>http://localhost:8000</code>
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

        process_btn.click(
            fn=process_orchestrator_request,
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
        
        **Note**: Some features require additional dependencies. Install with `pip install psutil aiohttp` to enable all features.
        """)
        
        with gr.Row():
            health_btn = gr.Button("Get Health Status", variant="primary")
            metrics_btn = gr.Button("Get Performance Metrics", variant="primary")
            cache_btn = gr.Button("Get Cache Status", variant="primary")
        
        health_output = gr.JSON(label="Health Status")
        metrics_output = gr.JSON(label="Performance Metrics")
        cache_output = gr.JSON(label="Cache Status")
        
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

# ----------------------------------------
# Main Entry Point
# ----------------------------------------
if __name__ == "__main__":
    demo.launch(
        mcp_server=True,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        share=False
    )