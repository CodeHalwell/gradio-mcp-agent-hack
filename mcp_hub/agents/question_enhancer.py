"""
Question Enhancer Agent module.

This module contains the QuestionEnhancerAgent class which is responsible for 
enhancing questions into sub-questions for research.
"""

from typing import Dict, Any

from ..config import api_config, model_config
from ..exceptions import ValidationError, APIError
from ..utils import validate_non_empty_string, extract_json_from_text, make_llm_completion
from ..logging_config import logger

# Import advanced features with graceful fallback
try:
    from ..performance_monitoring import track_performance
    from ..cache_utils import cached
    from ..reliability_utils import rate_limited, circuit_protected
except ImportError:
    # Create dummy decorators for backward compatibility
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
    
    def cached(ttl: int = 300):
        def decorator(func): 
            return func
        return decorator


class QuestionEnhancerAgent:
    """
    Agent responsible for enhancing questions into sub-questions for research.

    This agent takes a single user query and intelligently breaks it down into
    multiple distinct, non-overlapping sub-questions that explore different
    technical angles of the original request. It uses LLM models to enhance
    question comprehension and research depth.
    """
    
    @track_performance("question_enhancement")
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