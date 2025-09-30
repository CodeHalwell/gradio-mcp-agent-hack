"""Code Runner Agent for executing Python code in Modal sandboxes."""
import base64
import marshal
import types
import time
import asyncio
import textwrap
from typing import List
from contextlib import asynccontextmanager

import modal

from ..config import app_config
from ..exceptions import CodeExecutionError
from ..logging_config import logger
from ..decorators import with_performance_tracking, rate_limited


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
