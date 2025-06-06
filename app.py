"""Enhanced MCP Hub - Single Unified Version with Advanced Features."""

import gradio as gr
import modal
import textwrap
import base64
import marshal
import types
import time
from typing import Dict, Any, Optional
from functools import wraps

# Import our custom modules
from mcp_hub.config import api_config, model_config, app_config
from mcp_hub.exceptions import APIError, ValidationError, CodeGenerationError, CodeExecutionError
from mcp_hub.utils import (
    validate_non_empty_string, extract_json_from_text,
    extract_urls_from_text, make_nebius_completion,
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
    """Decorator to add performance tracking to any function."""
    def decorator(func):
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
    def enhance_question(self, user_request: str) -> Dict[str, Any]:
        """Split a single user query into three distinct sub-questions."""
        try:
            validate_non_empty_string(user_request, "User request")
            logger.info(f"Enhancing question: {user_request[:100]}...")
            
            prompt_text = f"""
            You are an AI assistant that must break a single user query into three distinct, non-overlapping sub-questions.
            Each sub-question should explore a different technical angle of the original request.
            Output must be valid JSON with a top-level key "sub_questions" whose value is an array of strings—no extra keys, no extra prose.

            User Request: "{user_request}"

            Respond with exactly:
            {{
            "sub_questions": [
                "First enhanced sub-question …",
                "Second enhanced sub-question …",
                "Third enhanced sub-question …"
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
            
            raw_output = make_nebius_completion(
                model=model_config.question_enhancer_model,
                messages=messages,
                temperature=0.0,
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
            
            output_text = make_nebius_completion(
                model=model_config.llm_processor_model,
                messages=messages,
                temperature=app_config.llm_temperature
            )
            
            logger.info(f"LLM processing completed for task: {task}")
            return {
                "input_text": text_input,
                "task": task,
                "provided_context": context,
                "llm_processed_output": output_text,
                "llm_model_used": model_config.llm_processor_model,
            }
            
        except (ValidationError, APIError) as e:
            logger.error(f"LLM processing failed: {str(e)}")
            return {"error": str(e), "input_text": text_input, "processed_output": None}
        except Exception as e:
            logger.error(f"Unexpected error in LLM processing: {str(e)}")
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

    @with_performance_tracking("code_generation")
    @rate_limited("nebius")
    @circuit_protected("nebius")
    def generate_code(
        self, user_request: str, grounded_context: str
    ) -> tuple[Dict[str, Any], str]:
        """Generate Python code based on user request and grounded context."""
        try:
            validate_non_empty_string(user_request, "User request")
            logger.info("Generating Python code")

            system_prompt = f"""
                    You are an expert Python developer. Given the user's request and the following
                    grounded context (search summaries and citations), generate a Python code
                    snippet that directly addresses the user's needs. Ensure the code is valid,
                    complete, and runnable.

                    User Request:
                    \"\"\"{user_request}\"\"\"

                    Grounded Context:
                    \"\"\"{grounded_context}\"\"\"

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

            for attempt in range(1, app_config.max_code_generation_attempts + 1):
                try:
                    logger.info("Code generation attempt %s", attempt)

                    messages = [{"role": "user", "content": system_prompt}]
                    raw_output = make_nebius_completion(
                        model=model_config.code_generator_model,
                        messages=messages,
                        temperature=app_config.code_gen_temperature,
                    )

                    # Validate that the code compiles
                    code_compiled = compile(raw_output, "<string>", "exec")

                    logger.info(f"The generated code is as follows: \n\n{raw_output}\n")
                    logger.info("Code generation successful")

                    return {"generated_code": code_compiled}, raw_output

                except SyntaxError as e:
                    logger.warning(
                        "Generated code syntax error (attempt %s): %s", attempt, e
                    )
                    if attempt == app_config.max_code_generation_attempts:
                        raise CodeGenerationError(
                            f"Failed to generate valid Python after {attempt} attempts"
                        )
                    # Otherwise retry
                    continue

                except APIError as e:
                    # Fatal — surface as CodeGenerationError
                    raise CodeGenerationError(f"Unexpected API error: {e}") from e

                except Exception as e:
                    logger.error(
                        "Code generation error (attempt %s): %s",
                        attempt,
                        e,
                    )
                    if attempt == app_config.max_code_generation_attempts:
                        raise CodeGenerationError(f"Unexpected error: {e}")
                    # Otherwise retry
                    continue

            # If the loop finishes without a successful return, raise explicitly
            raise CodeGenerationError("No valid code produced after all attempts")

        except (ValidationError, APIError, CodeGenerationError) as e:
            logger.error("Code generation failed: %s", e)
            return {"error": str(e), "generated_code": ""}, ""

        except Exception as e:
            logger.error("Unexpected error in code generation: %s", e)
            return {"error": f"Unexpected error: {e}", "generated_code": ""}, ""

class CodeRunnerAgent:
    """Agent responsible for running code in Modal sandbox."""
    
    def __init__(self):
        self.app = modal.App.lookup(app_config.modal_app_name, create_if_missing=True)
    
    @with_performance_tracking("code_execution")
    @rate_limited("modal")
    def run_code(self, code_or_obj) -> str:
        """Execute code in Modal sandbox."""
        try:
            logger.info("Executing code in Modal sandbox")
            
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
            
            sb = None
            try:
                sb = modal.Sandbox.create(app=self.app)
                proc = sb.exec("python", "-c", payload)
                output = proc.stdout.read() + proc.stderr.read()
                logger.info("Code execution completed successfully")
                return output
            finally:
                if sb:
                    try:
                        sb.terminate()
                    except Exception as e:
                        logger.warning(f"Failed to terminate sandbox: {str(e)}")
                        
        except CodeExecutionError:
            raise
        except Exception as e:
            logger.error(f"Code execution failed: {str(e)}")
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
            enhancer_result = self.question_enhancer.enhance_question(user_request)
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
                final_narrative = make_nebius_completion(
                    model=model_config.orchestrator_model,
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
    
    def _format_search_results(self, results: list) -> str:
        """Format search results into a readable string."""
        if not results:
            return "No search results found."
        
        snippets = []
        for idx, item in enumerate(results, 1):
            title = item.get("title", "No Title")
            url = item.get("url", "")
            content = item.get("content", "")
            
            snippet = f"Result {idx}:\nTitle: {title}\nURL: {url}\nSnippet: {content}\n"
            snippets.append(snippet)
        
        return "\n".join(snippets).strip()

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
        return health_monitor.get_health_status()
    except Exception as e:
        return {"error": f"Health monitoring failed: {str(e)}"}

def get_performance_metrics() -> Dict[str, Any]:
    """Get performance metrics and analytics."""
    if not ADVANCED_FEATURES_AVAILABLE:
        return {
            "status": "basic_mode", 
            "message": "Performance metrics not available. Install 'pip install psutil aiohttp' to enable advanced monitoring."
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
            "message": "Cache monitoring not available. Install 'pip install psutil aiohttp' to enable cache statistics."
        }
    
    try:
        from mcp_hub.cache_utils import cache_manager
        return cache_manager.get_cache_status()
    except Exception as e:
        return {"error": f"Cache status failed: {str(e)}"}

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
    return question_enhancer.enhance_question(user_request)

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

def get_health_status() -> Dict[str, Any]:
    """Get comprehensive system health status."""
    if not ADVANCED_FEATURES_AVAILABLE:
        return {"error": "Advanced features not available"}
    
    try:
        from mcp_hub.health_monitoring import HealthMonitor
        health_monitor = HealthMonitor()
        return health_monitor.get_health_status()
    except Exception as e:
        return {"error": f"Health monitoring failed: {str(e)}"}

def get_performance_metrics() -> Dict[str, Any]:
    """Get performance metrics and analytics."""
    if not ADVANCED_FEATURES_AVAILABLE:
        return {"error": "Advanced features not available"}
    try:
        from mcp_hub.performance_monitoring import metrics_collector
        return metrics_collector.get_metrics_summary()
    except Exception as e:
        return {"error": f"Performance metrics failed: {str(e)}"}

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
        return health_monitor.get_health_status()
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

# ----------------------------------------
# Gradio UI / MCP Server Setup
# ----------------------------------------
with gr.Blocks(title="Deep Research & Code Assistant Hub", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        ## Deep Research & Code Assistant Hub (Improved Version)

        **Enhanced Workflow**:
        1. Break the user's request into three enhanced sub-questions.
        2. Perform Tavily web search (top-3 results) for each sub-question.
        3. Summarize each set of snippets via Nebius LLM.
        4. Extract APA-style citations from each snippet block.
        5. Combine summaries into a grounded context.
        6. Generate a Python code snippet based on that context.
        7. Run the generated code in a Modal sandbox.
        8. Return the final summary, generated code, execution output, and citations.
        
        **Improvements**:
        - Better error handling and logging
        - Modular agent-based architecture        - Configuration management
        - Input validation
        - Graceful failure handling
        """
    )
    
    with gr.Tab("Orchestrator (Research → Code Workflow)"):
        gr.Markdown("## AI Research & Code Assistant")
        gr.Markdown("""
        **Workflow:** Splits into sub-questions → Tavily search & summarization → Generate Python code → Execute via Modal → Return results with citations
        """)
        
        with gr.Row():
            # Left column - Input and JSON output
            with gr.Column(scale=1):
                input_textbox = gr.Textbox(
                    label="Your High-Level Request",
                    lines=4,
                    placeholder="E.g. 'Write Python code to scrape the latest stock prices and plot a graph.'",
                    info="Describe what you want to research and code"
                )
                json_output = gr.JSON(
                    label="Complete Orchestrated Output",
                    container=True
                )
                
            # Right column - Clean natural language output
            with gr.Column(scale=1):
                clean_output = gr.Markdown(
                    label="Summary & Results",
                    value="*Your results will appear here in a clean, readable format...*"
                )        # Process button
        process_btn = gr.Button("🚀 Process Request", variant="primary", size="lg")
        
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
        
        # Connect the button to the processing function
        process_btn.click(
            fn=process_orchestrator_request,
            inputs=[input_textbox],
            outputs=[json_output, clean_output],
            api_name="agent_orchestrator_service"
        )

    with gr.Tab("Agent: Question Enhancer"):
        gr.Interface(
            fn=agent_question_enhancer,
            inputs=[
                gr.Textbox(
                    label="Original User Request",
                    lines=2,
                    placeholder="Enter your question to be split into 3 sub-questions…"
                )
            ],
            outputs=gr.JSON(label="Enhanced Sub-Questions"),
            title="Question Enhancer Agent",
            description="Splits a single user query into 3 distinct sub-questions using Qwen models.",
            api_name="agent_question_enhancer_service",
        )

    with gr.Tab("Agent: Web Search"):
        gr.Interface(
            fn=agent_web_search,
            inputs=[gr.Textbox(label="Search Query", placeholder="Enter search term…")],
            outputs=gr.JSON(label="Web Search Results (Tavily)"),
            title="Web Search Agent",
            description="Perform a Tavily web search with configurable result limits.",
            api_name="agent_web_search_service",
        )

    with gr.Tab("Agent: LLM Processor (Nebius)"):
        gr.Interface(
            fn=agent_llm_processor,
            inputs=[
                gr.Textbox(label="Text to Process", lines=5, placeholder="Enter text for the LLM…"),
                gr.Dropdown(
                    choices=["summarize", "reason", "extract_keywords"],
                    value="summarize",
                    label="LLM Task",
                ),
                gr.Textbox(label="Optional Context", lines=2, placeholder="Background info…"),
            ],
            outputs=gr.JSON(label="LLM Processed Output"),
            title="LLM Processing Agent",
            description="Use Meta-Llama models for text processing tasks.",
            api_name="agent_llm_processor_service",
        )

    with gr.Tab("Agent: Citation Formatter"):
        gr.Interface(
            fn=agent_citation_formatter,
            inputs=[
                gr.Textbox(
                    label="Text Block (with URLs)",
                    lines=5,
                    placeholder="Paste text containing URLs to generate APA citations…"
                )
            ],
            outputs=gr.JSON(label="Formatted APA-style Citations"),
            title="Citation Formatter Agent",
            description="Extracts URLs from text and returns APA-style citations.",
            api_name="agent_citation_formatter_service",
        )

    with gr.Tab("Agent: Code Generator"):
        gr.Interface(
            fn=agent_code_generator,
            inputs=[
                gr.Textbox(label="Original Request", lines=2, placeholder="Enter your high-level request…"),
                gr.Textbox(label="Grounded Context", lines=8, placeholder="Paste combined summaries + citations…")
            ],
            outputs=gr.JSON(label="Generated Python Code"),
            title="Code Generator Agent",
            description="Generates Python code using Qwen2.5-Coder models.",
            api_name="agent_code_generator_service",
        )

    with gr.Tab("Agent: Code Runner (Modal)"):
        gr.Interface(
            fn=code_runner_wrapper,
            inputs=[gr.Code(label="Python Code to Execute", language="python")],
            outputs=[gr.Textbox(label="Execution Output")],
            title="Code Runner Agent",
            description="Executes Python code in a Modal sandbox with improved error handling.",
            api_name="agent_code_runner_service",
        )

    # Advanced feature tabs (if available)
    if ADVANCED_FEATURES_AVAILABLE:
        with gr.Tab("🏥 Health Monitor"):
            gr.Interface(
                fn=get_health_status,
                inputs=[],
                outputs=gr.JSON(label="System Health Status"),
                title="System Health Monitor",
                description="Monitor system performance, API health, and resource usage.",
                api_name="health_monitor_service",
            )
        
        with gr.Tab("📊 Performance Analytics"):
            gr.Interface(
                fn=get_performance_metrics,
                inputs=[],
                outputs=gr.JSON(label="Performance Metrics"),
                title="Performance Analytics",
                description="View detailed performance metrics and analytics.",
                api_name="performance_metrics_service",
            )
        
        with gr.Tab("💾 Cache Status"):
            gr.Interface(
                fn=get_cache_status,
                inputs=[],
                outputs=gr.JSON(label="Cache Status & Statistics"),
                title="Cache Management",
                description="View cache statistics and status information.",
                api_name="cache_status_service",
            )
    else:
        with gr.Tab("ℹ️ Advanced Features"):
            gr.Markdown("""
            ### 🚀 Advanced Features Available
            
            Your MCP Hub can be enhanced with advanced monitoring and performance features:
            
            #### **📦 Installation:**
            ```bash
            pip install psutil aiohttp
            ```
            
            #### **✨ Features You'll Get:**
            
            🏥 **Health Monitor**: 
            - Real-time system health tracking
            - API connectivity monitoring
            - Resource usage analytics
            
            📊 **Performance Analytics**: 
            - Detailed operation metrics
            - Response time tracking
            - Success/failure rates
            
            💾 **Enhanced Caching**: 
            - Intelligent API call reduction
            - Faster response times
            - Cache statistics
            
            🛡️ **Reliability Features**: 
            - Rate limiting for API protection
            - Circuit breakers for fault tolerance
            - Automatic retry mechanisms
            
            #### **🔄 How to Enable:**
            1. Install the dependencies above
            2. Restart this application
            3. New tabs will appear with advanced features!
              **Note**: These features are completely optional. Your system works perfectly without them.
            """)

if __name__ == "__main__":
    if ADVANCED_FEATURES_AVAILABLE:
        logger.info("Launching Enhanced MCP Hub with Advanced Features...")
    else:
        logger.info("Launching MCP Hub (Basic Mode)...")
    
    logger.info("Ensure your .env has:\n  TAVILY_API_KEY=tvly-...\n  NEBIUS_API_KEY=nb-...")
    logger.info("Install dependencies: uv add gradio requests tavily-python openai python-dotenv modal")
    
    if ADVANCED_FEATURES_AVAILABLE:
        logger.info("Advanced features loaded:")
        logger.info("   - Performance monitoring")
        logger.info("   - Health monitoring") 
        logger.info("   - Intelligent caching")
        logger.info("   - Rate limiting & circuit breakers")
    else:
        logger.info("Running in basic mode. Install 'uv add psutil aiohttp' for advanced features.")
    
    try:
        demo.launch(mcp_server=True, server_name="127.0.0.1")
        logger.info("MCP schema available at http://127.0.0.1:7860/gradio_api/mcp/schema")
    except Exception as e:
        logger.error(f"Failed to launch application: {str(e)}")
        raise
