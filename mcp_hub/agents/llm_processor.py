"""LLMProcessorAgent - Processes text using LLMs for various tasks."""

from typing import Dict, Any
from mcp_hub.config import api_config, model_config, app_config
from mcp_hub.exceptions import ValidationError, APIError
from mcp_hub.utils import validate_non_empty_string, make_llm_completion
from mcp_hub.logging_config import logger

# Import decorators with graceful fallback
try:
    from mcp_hub.performance_monitoring import track_performance
    from mcp_hub.reliability_utils import rate_limited, circuit_protected
except ImportError:
    # Fallback decorators if advanced features not available
    def track_performance(operation_name: str = None):
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


class LLMProcessorAgent:
    """
    Agent responsible for processing text using Large Language Models for various tasks.

    This agent handles text processing operations including summarization, reasoning,
    and keyword extraction using configured LLM providers. It supports both synchronous
    and asynchronous processing with configurable temperature and response formats.
    """

    @track_performance(operation_name="llm_processing")
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

    @track_performance(operation_name="async_llm_processing")
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
"""LLM Processor Agent for text processing tasks using language models."""
from typing import Dict, Any

from ..config import api_config, model_config, app_config
from ..exceptions import APIError, ValidationError
from ..logging_config import logger
from ..utils import validate_non_empty_string, make_llm_completion
from ..decorators import with_performance_tracking, rate_limited, circuit_protected


class LLMProcessorAgent:
    """
    Agent responsible for processing text using Large Language Models for various tasks.

    This agent handles text processing operations including summarization, reasoning,
    and keyword extraction using configured LLM providers. It supports both synchronous
    and asynchronous processing with configurable temperature and response formats.
    """
    
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
