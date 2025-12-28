"""Enhanced input validation with security checks and limits."""

import re
from typing import Any
from mcp_hub.exceptions import ValidationError

# Configuration for validation limits
MAX_INPUT_LENGTH = 10000  # Maximum characters for user input
MAX_CONTEXT_LENGTH = 50000  # Maximum characters for context/grounded text
MIN_INPUT_LENGTH = 1  # Minimum non-whitespace characters

# Patterns for security checks
SUSPICIOUS_PATTERNS = [
    r'<script[^>]*>.*?</script>',  # Script tags
    r'javascript:',  # JavaScript protocol
    r'on\w+\s*=',  # Event handlers (onclick, onerror, etc.)
    r'data:text/html',  # Data URIs with HTML
    r'vbscript:',  # VBScript protocol
]


def validate_user_input(
    text: Any,
    field_name: str = "Input",
    max_length: int = MAX_INPUT_LENGTH,
    min_length: int = MIN_INPUT_LENGTH,
    allow_empty: bool = False
) -> str:
    """
    Validate and sanitize user input with comprehensive security checks.

    This function performs multiple levels of validation:
    - Type checking (must be string)
    - Length validation (min/max bounds)
    - Control character detection
    - XSS/injection pattern detection
    - Whitespace normalization

    Args:
        text: The input to validate (will be converted to string)
        field_name: Name of the field for error messages
        max_length: Maximum allowed length in characters
        min_length: Minimum required length (after stripping)
        allow_empty: Whether to allow empty strings

    Returns:
        str: The validated and normalized input string

    Raises:
        ValidationError: If validation fails for any reason

    Examples:
        >>> validate_user_input("Hello world", "message")
        'Hello world'
        >>> validate_user_input("", "message", allow_empty=True)
        ''
        >>> validate_user_input("x" * 20000, "message")
        ValidationError: message exceeds maximum length
    """
    # Type checking
    if not isinstance(text, str):
        if text is None:
            if allow_empty:
                return ""
            raise ValidationError(f"{field_name} must be a non-empty string, got None")
        # Try to convert to string
        try:
            text = str(text)
        except Exception as e:
            raise ValidationError(f"{field_name} must be a string, got {type(text).__name__}: {e}")

    # Length validation (before stripping to prevent bypass)
    if len(text) > max_length:
        raise ValidationError(
            f"{field_name} exceeds maximum length of {max_length} characters "
            f"(got {len(text)} characters)"
        )

    # Strip whitespace for meaningful content check
    stripped = text.strip()

    # Empty check
    if not stripped:
        if allow_empty:
            return stripped
        raise ValidationError(f"{field_name} must not be empty or whitespace-only")

    # Minimum length check (after stripping)
    if len(stripped) < min_length:
        raise ValidationError(
            f"{field_name} must be at least {min_length} characters "
            f"(got {len(stripped)} after stripping whitespace)"
        )

    # Control character check (except common ones: \n, \r, \t)
    if any(ord(c) < 32 and c not in '\n\r\t' for c in text):
        raise ValidationError(
            f"{field_name} contains invalid control characters. "
            "Only newlines, tabs, and carriage returns are allowed."
        )

    # Check for suspicious patterns (XSS, injection attempts)
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            raise ValidationError(
                f"{field_name} contains potentially malicious content. "
                "HTML tags, JavaScript, and scripts are not allowed."
            )

    # Check for NULL bytes
    if '\x00' in text:
        raise ValidationError(f"{field_name} contains NULL bytes which are not allowed")

    # Normalize line endings to \n
    normalized = text.replace('\r\n', '\n').replace('\r', '\n')

    return normalized


def validate_code_input(code: str, max_length: int = 50000) -> str:
    """
    Validate code input with relaxed rules compared to user text.

    Code can contain more characters and patterns, but still needs basic validation.

    Args:
        code: The code string to validate
        max_length: Maximum allowed length

    Returns:
        str: The validated code string

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(code, str):
        raise ValidationError(f"Code must be a string, got {type(code).__name__}")

    if len(code) > max_length:
        raise ValidationError(
            f"Code exceeds maximum length of {max_length} characters "
            f"(got {len(code)} characters)"
        )

    if not code.strip():
        raise ValidationError("Code must not be empty")

    # Check for NULL bytes
    if '\x00' in code:
        raise ValidationError("Code contains NULL bytes which are not allowed")

    return code


def validate_json_input(text: str) -> str:
    """
    Validate input expected to be JSON.

    Args:
        text: The JSON string to validate

    Returns:
        str: The validated JSON string

    Raises:
        ValidationError: If validation fails
    """
    import json

    validated = validate_user_input(
        text,
        field_name="JSON input",
        max_length=100000,  # JSON can be larger
        allow_empty=False
    )

    # Try to parse as JSON
    try:
        json.loads(validated)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON format: {str(e)}")

    return validated


def sanitize_error_message(error: Exception) -> str:
    """
    Sanitize error messages to prevent information leakage.

    Removes sensitive information like:
    - File system paths
    - API keys or tokens
    - Internal IP addresses
    - Stack traces

    Args:
        error: The exception to sanitize

    Returns:
        str: A sanitized error message safe for display
    """
    message = str(error)

    # Remove file system paths
    message = re.sub(r'/home/[\w/]+/', '/app/', message)
    message = re.sub(r'C:\\[^\\]+\\', 'C:\\app\\', message)
    message = re.sub(r'/usr/[^/]+/[^/]+/', '/usr/local/', message)

    # Remove API keys (various formats)
    message = re.sub(
        r'(api[_-]?key|token|secret|password)["\']?\s*[:=]\s*["\']?[\w-]+',
        r'\1=***',
        message,
        flags=re.IGNORECASE
    )

    # Remove bearer tokens
    message = re.sub(r'Bearer\s+[\w-]+', 'Bearer ***', message, flags=re.IGNORECASE)

    # Remove IP addresses (simple pattern)
    message = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '***.***.***. ***', message)

    # Truncate very long messages
    if len(message) > 500:
        message = message[:500] + '... (truncated)'

    return message


def validate_model_name(model_name: str) -> str:
    """
    Validate model name to prevent injection attacks.

    Args:
        model_name: The model name to validate

    Returns:
        str: The validated model name

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(model_name, str):
        raise ValidationError(f"Model name must be a string, got {type(model_name).__name__}")

    if not model_name.strip():
        raise ValidationError("Model name must not be empty")

    # Model names should only contain alphanumeric, hyphens, underscores, slashes, dots
    if not re.match(r'^[a-zA-Z0-9/_.-]+$', model_name):
        raise ValidationError(
            "Model name contains invalid characters. "
            "Only alphanumeric characters, hyphens, underscores, slashes, and dots are allowed."
        )

    if len(model_name) > 200:
        raise ValidationError(f"Model name too long (max 200 characters)")

    return model_name


def validate_url(url: str) -> str:
    """
    Validate URL format and security.

    Args:
        url: The URL to validate

    Returns:
        str: The validated URL

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(url, str):
        raise ValidationError(f"URL must be a string, got {type(url).__name__}")

    if not url.strip():
        raise ValidationError("URL must not be empty")

    # Basic URL pattern
    url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    if not re.match(url_pattern, url, re.IGNORECASE):
        raise ValidationError("Invalid URL format. Must start with http:// or https://")

    # Check for suspicious patterns
    suspicious_url_patterns = [
        r'javascript:',
        r'data:',
        r'file:',
        r'vbscript:',
    ]

    for pattern in suspicious_url_patterns:
        if re.search(pattern, url, re.IGNORECASE):
            raise ValidationError(f"URL contains suspicious protocol: {pattern}")

    if len(url) > 2048:
        raise ValidationError("URL too long (max 2048 characters)")

    return url
