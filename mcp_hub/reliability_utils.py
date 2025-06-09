"""Rate limiting and circuit breaker patterns for robust API interactions."""

import time
from datetime import datetime
from typing import Callable, Any, Dict
from functools import wraps
from threading import Lock
from collections import deque
from .exceptions import APIError
from .logging_config import logger

class RateLimiter:
    """Token bucket rate limiter for API calls."""
    
    def __init__(self, calls_per_second: float = 1.0, burst_size: int = 5):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_second: Maximum calls per second
            burst_size: Maximum burst of calls allowed
        """
        self.calls_per_second = calls_per_second
        self.burst_size = float(burst_size)
        self.tokens = float(burst_size)
        self.last_update = time.time()
        self.lock = Lock()
    
    def acquire(self, timeout: float = None) -> bool:
        """
        Acquire a token for making an API call.
        
        Args:
            timeout: Maximum time to wait for a token
            
        Returns:
            True if token acquired, False if timeout
        """
        start_time = time.time()
        
        while True:
            with self.lock:
                now = time.time()
                # Add tokens based on elapsed time
                time_passed = now - self.last_update
                self.tokens = min(
                    self.burst_size,
                    self.tokens + time_passed * self.calls_per_second
                )
                self.last_update = now
                
                if self.tokens >= 1:
                    self.tokens -= 1
                    return True
            
            # Check timeout
            if timeout and (time.time() - start_time) >= timeout:
                return False
            
            # Wait before retrying
            time.sleep(0.1)

class CircuitBreaker:
    """Circuit breaker pattern for handling API failures gracefully."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before trying again
            expected_exception: Exception type that triggers circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = Lock()
    
    def _can_attempt(self) -> bool:
        """Check if we can attempt the operation."""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if (datetime.now() - self.last_failure_time).seconds >= self.timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def _record_success(self):
        """Record a successful operation."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments for the function
            
        Returns:
            Function result
            
        Raises:
            APIError: If circuit is open or function fails
        """
        with self.lock:
            if not self._can_attempt():
                raise APIError(
                    "CircuitBreaker",
                    f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}"
                )
        
        try:
            result = func(*args, **kwargs)
            with self.lock:
                self._record_success()
            return result
            
        except self.expected_exception as e:
            with self.lock:
                self._record_failure()
            logger.error(f"Circuit breaker recorded failure: {str(e)}")
            raise APIError("CircuitBreaker", f"Protected function failed: {str(e)}")

# Global instances for different services
nebius_rate_limiter = RateLimiter(calls_per_second=2.0, burst_size=5)
tavily_rate_limiter = RateLimiter(calls_per_second=1.0, burst_size=3)

nebius_circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=30)
tavily_circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=30)

def rate_limited(service: str = "default", timeout: float = 10.0):
    """
    Decorator to rate limit function calls.
    
    Args:
        service: Service name (nebius, tavily, or default)
        timeout: Maximum time to wait for rate limit token
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Select appropriate rate limiter
            if service == "nebius":
                limiter = nebius_rate_limiter
            elif service == "tavily":
                limiter = tavily_rate_limiter
            else:
                limiter = RateLimiter()  # Default limiter
            
            if not limiter.acquire(timeout=timeout):
                raise APIError(service, f"Rate limit timeout after {timeout}s")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def circuit_protected(service: str = "default"):
    """
    Decorator to protect function calls with circuit breaker.
    
    Args:
        service: Service name (nebius, tavily, or default)
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Select appropriate circuit breaker
            if service == "nebius":
                breaker = nebius_circuit_breaker
            elif service == "tavily":
                breaker = tavily_circuit_breaker
            else:
                breaker = CircuitBreaker()  # Default breaker
            
            return breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator

class APIHealthMonitor:
    """Monitor API health and performance metrics."""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize health monitor.
        
        Args:
            window_size: Number of recent calls to track
        """
        self.window_size = window_size
        self.call_history = deque(maxlen=window_size)
        self.lock = Lock()
    
    def record_call(self, service: str, success: bool, response_time: float):
        """Record an API call result."""
        with self.lock:
            self.call_history.append({
                "service": service,
                "success": success,
                "response_time": response_time,
                "timestamp": datetime.now()
            })
    
    def get_health_stats(self, service: str = None) -> Dict[str, Any]:
        """Get health statistics for a service or all services."""
        with self.lock:
            if service:
                calls = [call for call in self.call_history if call["service"] == service]
            else:
                calls = list(self.call_history)
        
        if not calls:
            return {"error": "No call history available"}
        
        total_calls = len(calls)
        successful_calls = sum(1 for call in calls if call["success"])
        success_rate = successful_calls / total_calls
        
        response_times = [call["response_time"] for call in calls]
        avg_response_time = sum(response_times) / len(response_times)
        
        return {
            "service": service or "all",
            "total_calls": total_calls,
            "success_rate": success_rate,
            "avg_response_time_ms": avg_response_time * 1000,
            "recent_failures": total_calls - successful_calls
        }

# Global health monitor
health_monitor = APIHealthMonitor()
