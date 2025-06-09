"""Custom exception classes for the MCP Hub project."""

class MCPHubError(Exception):
    """Base exception class for MCP Hub errors."""
    pass

class APIError(MCPHubError):
    """Raised when API calls fail."""
    def __init__(self, service: str, message: str):
        self.service = service
        self.message = message
        super().__init__(f"{service} API Error: {message}")

class ConfigurationError(MCPHubError):
    """Raised when there are configuration issues."""
    pass

class ValidationError(MCPHubError):
    """Raised when input validation fails."""
    pass

class CodeGenerationError(MCPHubError):
    """Raised when code generation fails."""
    pass

class CodeExecutionError(MCPHubError):
    """Raised when code execution fails."""
    pass
