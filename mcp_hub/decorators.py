"""Performance tracking and monitoring decorators for MCP Hub agents."""
import time
import asyncio
from functools import wraps
from typing import Callable

from .logging_config import logger

# Import advanced features with graceful fallback
ADVANCED_FEATURES_AVAILABLE = False
try:
    from .performance_monitoring import metrics_collector
    from .cache_utils import cached
    from .reliability_utils import rate_limited, circuit_protected
    ADVANCED_FEATURES_AVAILABLE = True
    logger.info("Advanced features loaded in decorators module")
except ImportError as e:
    logger.info(f"Advanced features not available in decorators: {e}")
    
    # Create dummy decorators for backward compatibility
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


def with_performance_tracking(operation_name: str) -> Callable:
    """
    Add performance tracking and metrics collection to any function (sync or async).

    This decorator wraps both synchronous and asynchronous functions to collect
    execution time, success/failure metrics, and error counts. It integrates with
    the advanced monitoring system when available.

    Args:
        operation_name (str): The name of the operation to track in metrics

    Returns:
        Callable: A decorator function that can wrap sync or async functions
    """
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
