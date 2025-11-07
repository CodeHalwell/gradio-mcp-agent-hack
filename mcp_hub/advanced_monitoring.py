"""Advanced performance monitoring and profiling.

This module provides comprehensive performance monitoring capabilities including:
- Request/response logging with detailed metrics
- Slow query detection and alerting
- Memory profiling
- Performance bottleneck detection
- Distributed tracing support
- Advanced metrics aggregation
"""

import time
import psutil
import threading
import tracemalloc
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from contextlib import contextmanager
from .logging_config import logger

# Try to import memory_profiler for advanced profiling
try:
    from memory_profiler import profile as memory_profile
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    logger.warning("memory_profiler not available. Install with: pip install memory-profiler")


@dataclass
class RequestTrace:
    """Detailed trace of a single request."""

    trace_id: str
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status: str = "running"  # running, success, failed
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    spans: List['Span'] = field(default_factory=list)

    def complete(self, status: str = "success", error: Optional[str] = None):
        """Complete the trace."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = status
        self.error = error

    def add_span(self, name: str, duration: float, metadata: Optional[Dict] = None):
        """Add a span (sub-operation) to the trace."""
        span = Span(
            name=name,
            start_time=time.time() - duration,
            duration=duration,
            metadata=metadata or {}
        )
        self.spans.append(span)

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary."""
        return {
            "trace_id": self.trace_id,
            "operation": self.operation,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "status": self.status,
            "error": self.error,
            "metadata": self.metadata,
            "spans": [span.to_dict() for span in self.spans]
        }


@dataclass
class Span:
    """A span represents a sub-operation within a trace."""

    name: str
    start_time: float
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary."""
        return {
            "name": self.name,
            "start_time": self.start_time,
            "duration": self.duration,
            "metadata": self.metadata
        }


@dataclass
class SlowQuery:
    """Record of a slow operation."""

    operation: str
    duration: float
    timestamp: datetime
    trace_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation": self.operation,
            "duration": self.duration,
            "timestamp": self.timestamp.isoformat(),
            "trace_id": self.trace_id,
            "metadata": self.metadata
        }


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""

    timestamp: datetime
    rss_mb: float
    vms_mb: float
    percent: float
    available_mb: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "rss_mb": self.rss_mb,
            "vms_mb": self.vms_mb,
            "percent": self.percent,
            "available_mb": self.available_mb
        }


class AdvancedMonitor:
    """Advanced monitoring and profiling system."""

    def __init__(
        self,
        slow_query_threshold: float = 5.0,
        max_traces: int = 1000,
        max_slow_queries: int = 100,
        memory_tracking_enabled: bool = True
    ):
        """
        Initialize advanced monitor.

        Args:
            slow_query_threshold: Threshold in seconds for slow query detection
            max_traces: Maximum number of traces to keep in memory
            max_slow_queries: Maximum number of slow queries to keep
            memory_tracking_enabled: Enable memory tracking
        """
        self.slow_query_threshold = slow_query_threshold
        self.max_traces = max_traces
        self.max_slow_queries = max_slow_queries
        self.memory_tracking_enabled = memory_tracking_enabled

        # Storage
        self.traces = deque(maxlen=max_traces)
        self.slow_queries = deque(maxlen=max_slow_queries)
        self.memory_snapshots = deque(maxlen=1000)

        # Active traces (keyed by trace_id)
        self.active_traces: Dict[str, RequestTrace] = {}

        # Metrics
        self.operation_stats = defaultdict(lambda: {
            "count": 0,
            "total_duration": 0.0,
            "min_duration": float('inf'),
            "max_duration": 0.0,
            "errors": 0
        })

        # Lock for thread safety
        self.lock = threading.Lock()

        # Start memory tracking thread if enabled
        if self.memory_tracking_enabled:
            self.memory_thread = threading.Thread(
                target=self._track_memory,
                daemon=True
            )
            self.memory_thread.start()

        logger.info("Advanced monitoring initialized")

    def _track_memory(self):
        """Background thread for tracking memory usage."""
        while True:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                vm = psutil.virtual_memory()

                snapshot = MemorySnapshot(
                    timestamp=datetime.now(),
                    rss_mb=memory_info.rss / (1024 * 1024),
                    vms_mb=memory_info.vms / (1024 * 1024),
                    percent=vm.percent,
                    available_mb=vm.available / (1024 * 1024)
                )

                with self.lock:
                    self.memory_snapshots.append(snapshot)

                time.sleep(30)  # Track every 30 seconds

            except Exception as e:
                logger.error(f"Error tracking memory: {e}")
                time.sleep(60)

    @contextmanager
    def trace_request(self, operation: str, metadata: Optional[Dict] = None):
        """Context manager for tracing a request.

        Usage:
            with monitor.trace_request('web_search', {'query': 'python'}):
                # Perform operation
                pass
        """
        import uuid
        trace_id = str(uuid.uuid4())

        trace = RequestTrace(
            trace_id=trace_id,
            operation=operation,
            start_time=time.time(),
            metadata=metadata or {}
        )

        with self.lock:
            self.active_traces[trace_id] = trace

        try:
            yield trace
            trace.complete(status="success")

            # Check if slow query
            if trace.duration and trace.duration > self.slow_query_threshold:
                slow_query = SlowQuery(
                    operation=operation,
                    duration=trace.duration,
                    timestamp=datetime.now(),
                    trace_id=trace_id,
                    metadata=metadata or {}
                )
                with self.lock:
                    self.slow_queries.append(slow_query)
                logger.warning(
                    f"Slow query detected: {operation} took {trace.duration:.2f}s"
                )

        except Exception as e:
            trace.complete(status="failed", error=str(e))
            raise

        finally:
            with self.lock:
                # Move from active to completed
                self.active_traces.pop(trace_id, None)
                self.traces.append(trace)

                # Update operation stats
                stats = self.operation_stats[operation]
                stats["count"] += 1
                if trace.duration:
                    stats["total_duration"] += trace.duration
                    stats["min_duration"] = min(stats["min_duration"], trace.duration)
                    stats["max_duration"] = max(stats["max_duration"], trace.duration)
                if trace.status == "failed":
                    stats["errors"] += 1

    @contextmanager
    def trace_span(self, trace: RequestTrace, span_name: str, metadata: Optional[Dict] = None):
        """Context manager for tracing a span within a request.

        Usage:
            with monitor.trace_request('orchestrate') as trace:
                with monitor.trace_span(trace, 'search'):
                    # Perform search
                    pass
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            trace.add_span(span_name, duration, metadata)

    def get_recent_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent request traces.

        Args:
            limit: Maximum number of traces to return

        Returns:
            List of trace dictionaries
        """
        with self.lock:
            traces = list(self.traces)[-limit:]
            return [trace.to_dict() for trace in reversed(traces)]

    def get_active_traces(self) -> List[Dict[str, Any]]:
        """Get currently active traces.

        Returns:
            List of active trace dictionaries
        """
        with self.lock:
            return [trace.to_dict() for trace in self.active_traces.values()]

    def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent slow queries.

        Args:
            limit: Maximum number to return

        Returns:
            List of slow query dictionaries
        """
        with self.lock:
            queries = list(self.slow_queries)[-limit:]
            return [q.to_dict() for q in reversed(queries)]

    def get_operation_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics for all operations.

        Returns:
            Dictionary of operation statistics
        """
        with self.lock:
            stats = {}
            for operation, op_stats in self.operation_stats.items():
                count = op_stats["count"]
                if count > 0:
                    avg_duration = op_stats["total_duration"] / count
                    error_rate = op_stats["errors"] / count

                    stats[operation] = {
                        "count": count,
                        "average_duration": avg_duration,
                        "min_duration": op_stats["min_duration"],
                        "max_duration": op_stats["max_duration"],
                        "total_duration": op_stats["total_duration"],
                        "errors": op_stats["errors"],
                        "error_rate": error_rate
                    }

            return stats

    def get_memory_stats(self, minutes: int = 5) -> Dict[str, Any]:
        """Get memory statistics for the last N minutes.

        Args:
            minutes: Number of minutes to analyze

        Returns:
            Dictionary of memory statistics
        """
        with self.lock:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            recent_snapshots = [
                s for s in self.memory_snapshots
                if s.timestamp >= cutoff_time
            ]

            if not recent_snapshots:
                return {"error": "No memory data available"}

            rss_values = [s.rss_mb for s in recent_snapshots]
            percent_values = [s.percent for s in recent_snapshots]

            return {
                "current_rss_mb": recent_snapshots[-1].rss_mb,
                "current_percent": recent_snapshots[-1].percent,
                "average_rss_mb": sum(rss_values) / len(rss_values),
                "average_percent": sum(percent_values) / len(percent_values),
                "max_rss_mb": max(rss_values),
                "max_percent": max(percent_values),
                "min_rss_mb": min(rss_values),
                "min_percent": min(percent_values),
                "snapshots_count": len(recent_snapshots),
                "time_window_minutes": minutes
            }

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report.

        Returns:
            Dictionary with full performance metrics
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "operation_stats": self.get_operation_stats(),
            "memory_stats": self.get_memory_stats(minutes=5),
            "slow_queries": self.get_slow_queries(limit=10),
            "active_traces_count": len(self.active_traces),
            "total_traces_count": len(self.traces),
            "recent_traces": self.get_recent_traces(limit=5)
        }

    def detect_bottlenecks(self) -> List[Dict[str, Any]]:
        """Detect performance bottlenecks.

        Returns:
            List of detected bottlenecks with details
        """
        bottlenecks = []

        # Check for slow operations
        op_stats = self.get_operation_stats()
        for operation, stats in op_stats.items():
            if stats["average_duration"] > self.slow_query_threshold:
                bottlenecks.append({
                    "type": "slow_operation",
                    "operation": operation,
                    "average_duration": stats["average_duration"],
                    "count": stats["count"],
                    "recommendation": f"Consider optimizing {operation}"
                })

        # Check for high error rates
        for operation, stats in op_stats.items():
            if stats["error_rate"] > 0.1:  # 10% error rate
                bottlenecks.append({
                    "type": "high_error_rate",
                    "operation": operation,
                    "error_rate": stats["error_rate"],
                    "errors": stats["errors"],
                    "recommendation": f"Investigate errors in {operation}"
                })

        # Check memory usage
        memory_stats = self.get_memory_stats(minutes=5)
        if not memory_stats.get("error") and memory_stats["current_percent"] > 80:
            bottlenecks.append({
                "type": "high_memory_usage",
                "current_percent": memory_stats["current_percent"],
                "current_rss_mb": memory_stats["current_rss_mb"],
                "recommendation": "Memory usage is high, consider optimization"
            })

        return bottlenecks

    def reset_stats(self):
        """Reset all statistics (useful for testing or periodic resets)."""
        with self.lock:
            self.operation_stats.clear()
            self.traces.clear()
            self.slow_queries.clear()
            logger.info("Advanced monitoring statistics reset")


# Global advanced monitor instance
advanced_monitor = AdvancedMonitor(
    slow_query_threshold=5.0,
    max_traces=1000,
    max_slow_queries=100
)


# Decorator for automatic request tracing
def trace_operation(operation_name: Optional[str] = None):
    """Decorator to automatically trace an operation.

    Args:
        operation_name: Name of the operation (defaults to function name)

    Usage:
        @trace_operation("my_operation")
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or func.__name__

        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with advanced_monitor.trace_request(op_name):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with advanced_monitor.trace_request(op_name):
                    return func(*args, **kwargs)
            return sync_wrapper

    return decorator


# Import asyncio after defining the decorator
import asyncio
from functools import wraps
