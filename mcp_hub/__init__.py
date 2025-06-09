"""MCP Hub - Multi-Agent Communication Protocol Hub for Research and Code Generation."""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Advanced MCP Hub with intelligent agent orchestration"

# Core imports that should be available at package level
try:
    from .config import api_config, model_config, app_config
    from .exceptions import APIError, ValidationError, CodeGenerationError, CodeExecutionError
    from .logging_config import logger
    
    __all__ = [
        "api_config", "model_config", "app_config",
        "APIError", "ValidationError", "CodeGenerationError", "CodeExecutionError",
        "logger"
    ]
except ImportError:
    # Graceful degradation for missing dependencies
    __all__ = []
