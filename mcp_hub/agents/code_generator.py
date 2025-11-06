"""CodeGeneratorAgent - Generates secure Python code with validation."""

import ast
from typing import Dict, Any
from mcp_hub.config import api_config, model_config, app_config
from mcp_hub.exceptions import ValidationError, APIError, CodeGenerationError
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

    @track_performance(operation_name="code_generation")
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
