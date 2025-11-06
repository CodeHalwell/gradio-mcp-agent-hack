"""Retry utilities with exponential backoff for resilient API calls.

Provides decorators and functions for retrying failed operations with
configurable backoff strategies, timeout handling, and error filtering.
"""

import time
import asyncio
import functools
from typing import Callable, Type, Tuple, Optional
from mcp_hub.logging_config import logger


def exponential_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
) -> float:
    """
    Calculate exponential backoff delay with optional jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Whether to add random jitter (recommended)

    Returns:
        float: Delay in seconds before next retry

    Example:
        >>> exponential_backoff(0)  # First retry
        ~1.0 seconds
        >>> exponential_backoff(1)  # Second retry
        ~2.0 seconds
        >>> exponential_backoff(2)  # Third retry
        ~4.0 seconds
    """
    delay = min(base_delay * (exponential_base ** attempt), max_delay)

    if jitter:
        # Add jitter to prevent thundering herd
        import random
        delay = delay * (0.5 + random.random() * 0.5)

    return delay


def retry_sync(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None
):
    """
    Decorator for retrying synchronous functions with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts (including initial call)
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exception types to retry on
        on_retry: Optional callback function called on each retry

    Returns:
        Decorated function with retry logic

    Example:
        @retry_sync(max_attempts=3, base_delay=2.0)
        def call_api():
            return requests.get("https://api.example.com")
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts - 1:
                        # Last attempt failed, raise the exception
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {str(e)}"
                        )
                        raise

                    # Calculate delay for next retry
                    delay = exponential_backoff(
                        attempt,
                        base_delay=base_delay,
                        max_delay=max_delay,
                        exponential_base=exponential_base
                    )

                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_attempts} failed: {str(e)}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    # Call retry callback if provided
                    if on_retry:
                        try:
                            on_retry(attempt, e, delay)
                        except Exception as callback_error:
                            logger.warning(f"Retry callback failed: {callback_error}")

                    # Wait before next retry
                    time.sleep(delay)

            # This should never be reached, but just in case
            raise last_exception

        return wrapper
    return decorator


def retry_async(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None
):
    """
    Decorator for retrying asynchronous functions with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts (including initial call)
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exception types to retry on
        on_retry: Optional callback function called on each retry

    Returns:
        Decorated async function with retry logic

    Example:
        @retry_async(max_attempts=3, base_delay=2.0)
        async def call_api():
            async with aiohttp.ClientSession() as session:
                return await session.get("https://api.example.com")
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts - 1:
                        # Last attempt failed, raise the exception
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {str(e)}"
                        )
                        raise

                    # Calculate delay for next retry
                    delay = exponential_backoff(
                        attempt,
                        base_delay=base_delay,
                        max_delay=max_delay,
                        exponential_base=exponential_base
                    )

                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_attempts} failed: {str(e)}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    # Call retry callback if provided
                    if on_retry:
                        try:
                            if asyncio.iscoroutinefunction(on_retry):
                                await on_retry(attempt, e, delay)
                            else:
                                on_retry(attempt, e, delay)
                        except Exception as callback_error:
                            logger.warning(f"Retry callback failed: {callback_error}")

                    # Wait before next retry
                    await asyncio.sleep(delay)

            # This should never be reached, but just in case
            raise last_exception

        return wrapper
    return decorator


class RetryConfig:
    """Configuration for retry behavior."""

    # LLM API retry configuration
    LLM_API = {
        'max_attempts': 3,
        'base_delay': 2.0,
        'max_delay': 30.0,
        'exponential_base': 2.0
    }

    # Search API retry configuration
    SEARCH_API = {
        'max_attempts': 3,
        'base_delay': 1.0,
        'max_delay': 10.0,
        'exponential_base': 2.0
    }

    # Code execution retry configuration
    CODE_EXECUTION = {
        'max_attempts': 2,
        'base_delay': 5.0,
        'max_delay': 30.0,
        'exponential_base': 2.0
    }

    # Network operation retry configuration
    NETWORK = {
        'max_attempts': 4,
        'base_delay': 1.0,
        'max_delay': 20.0,
        'exponential_base': 2.0
    }


def should_retry_exception(exception: Exception) -> bool:
    """
    Determine if an exception should trigger a retry.

    Args:
        exception: The exception to check

    Returns:
        bool: True if the exception is retryable, False otherwise
    """
    # Retryable exceptions (typically transient errors)
    retryable_exceptions = (
        ConnectionError,
        TimeoutError,
        OSError,
    )

    # Check if it's a known retryable exception
    if isinstance(exception, retryable_exceptions):
        return True

    # Check exception message for common transient error patterns
    error_msg = str(exception).lower()
    transient_patterns = [
        'timeout',
        'timed out',
        'connection',
        'temporarily unavailable',
        'rate limit',
        'too many requests',
        '429',
        '503',
        '504',
        'gateway timeout',
        'service unavailable',
    ]

    for pattern in transient_patterns:
        if pattern in error_msg:
            return True

    return False


def retry_with_filter(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exception_filter: Optional[Callable[[Exception], bool]] = None
):
    """
    Smart retry decorator that filters which exceptions to retry.

    Only retries on transient errors, not on permanent failures.

    Args:
        max_attempts: Maximum number of attempts
        base_delay: Initial delay between retries
        exception_filter: Optional function to determine if exception is retryable

    Example:
        @retry_with_filter(max_attempts=3)
        def call_api():
            # Will retry on connection errors, timeouts, etc.
            # Will NOT retry on 404, validation errors, etc.
            return api_client.fetch_data()
    """
    if exception_filter is None:
        exception_filter = should_retry_exception

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # Check if this exception should be retried
                    if not exception_filter(e):
                        logger.info(
                            f"{func.__name__} raised non-retryable exception: {type(e).__name__}: {str(e)}"
                        )
                        raise

                    if attempt == max_attempts - 1:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {str(e)}"
                        )
                        raise

                    delay = exponential_backoff(attempt, base_delay=base_delay)

                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_attempts} failed with retryable error: {str(e)}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    time.sleep(delay)

            raise last_exception

        return wrapper
    return decorator
