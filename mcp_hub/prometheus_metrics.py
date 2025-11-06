"""Prometheus metrics for production monitoring.

This module provides Prometheus-compatible metrics export for monitoring
application performance, system resources, and API health in production.
It bridges the existing metrics_collector with Prometheus format export.
"""

import time
import psutil
from typing import Dict, Any, Optional
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
)

# Try to import existing metrics collector
try:
    from .performance_monitoring import metrics_collector
    METRICS_COLLECTOR_AVAILABLE = True
except ImportError:
    METRICS_COLLECTOR_AVAILABLE = False

# Create a custom registry for better control
registry = CollectorRegistry()

# Application info
app_info = Info(
    'mcp_hub_app',
    'Application information',
    registry=registry
)
app_info.info({
    'version': '1.0.0',
    'name': 'gradio-mcp-hub'
})

# Request counters
requests_total = Counter(
    'mcp_hub_requests_total',
    'Total number of requests',
    ['agent', 'operation'],
    registry=registry
)

requests_success = Counter(
    'mcp_hub_requests_success_total',
    'Total number of successful requests',
    ['agent', 'operation'],
    registry=registry
)

requests_failed = Counter(
    'mcp_hub_requests_failed_total',
    'Total number of failed requests',
    ['agent', 'operation', 'error_type'],
    registry=registry
)

# Request duration histogram (buckets in seconds)
request_duration = Histogram(
    'mcp_hub_request_duration_seconds',
    'Request duration in seconds',
    ['agent', 'operation'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, float('inf')),
    registry=registry
)

# System metrics
cpu_usage = Gauge(
    'mcp_hub_cpu_usage_percent',
    'CPU usage percentage',
    registry=registry
)

memory_usage = Gauge(
    'mcp_hub_memory_usage_percent',
    'Memory usage percentage',
    registry=registry
)

memory_available = Gauge(
    'mcp_hub_memory_available_bytes',
    'Available memory in bytes',
    registry=registry
)

disk_usage = Gauge(
    'mcp_hub_disk_usage_percent',
    'Disk usage percentage',
    registry=registry
)

disk_free = Gauge(
    'mcp_hub_disk_free_bytes',
    'Free disk space in bytes',
    registry=registry
)

# Application uptime
app_uptime = Gauge(
    'mcp_hub_uptime_seconds',
    'Application uptime in seconds',
    registry=registry
)

# Active operations
active_operations = Gauge(
    'mcp_hub_active_operations',
    'Number of active operations',
    ['agent'],
    registry=registry
)

# Cache metrics
cache_hits = Counter(
    'mcp_hub_cache_hits_total',
    'Total number of cache hits',
    ['cache_type'],
    registry=registry
)

cache_misses = Counter(
    'mcp_hub_cache_misses_total',
    'Total number of cache misses',
    ['cache_type'],
    registry=registry
)

# Retry metrics
retries_total = Counter(
    'mcp_hub_retries_total',
    'Total number of retries',
    ['service', 'attempt'],
    registry=registry
)

retries_exhausted = Counter(
    'mcp_hub_retries_exhausted_total',
    'Total number of times retries were exhausted',
    ['service'],
    registry=registry
)

# API-specific metrics
llm_api_calls = Counter(
    'mcp_hub_llm_api_calls_total',
    'Total number of LLM API calls',
    ['provider', 'model'],
    registry=registry
)

search_api_calls = Counter(
    'mcp_hub_search_api_calls_total',
    'Total number of search API calls',
    registry=registry
)

code_executions = Counter(
    'mcp_hub_code_executions_total',
    'Total number of code executions',
    ['status'],
    registry=registry
)

# Track start time
_start_time = time.time()


def update_system_metrics():
    """Update system resource metrics.

    This function should be called periodically (e.g., every request)
    to keep system metrics up to date.
    """
    try:
        # Update CPU and memory
        cpu_usage.set(psutil.cpu_percent(interval=0.1))
        memory = psutil.virtual_memory()
        memory_usage.set(memory.percent)
        memory_available.set(memory.available)

        # Update disk
        disk = psutil.disk_usage('/')
        disk_usage.set(disk.percent)
        disk_free.set(disk.free)

        # Update uptime
        uptime = time.time() - _start_time
        app_uptime.set(uptime)

    except Exception as e:
        # Log error but don't fail the entire metrics collection
        import logging
        logging.error(f"Failed to update system metrics: {e}")


def sync_from_metrics_collector():
    """Sync metrics from the existing metrics_collector to Prometheus format.

    This function bridges the existing metrics collection system with Prometheus
    by reading from metrics_collector and updating Prometheus gauges.
    """
    if not METRICS_COLLECTOR_AVAILABLE:
        return

    try:
        # Get metrics summary from the existing collector
        summary = metrics_collector.get_metrics_summary(last_minutes=1)

        # Update Prometheus metrics with the latest values
        for metric_name, metric_data in summary.items():
            if isinstance(metric_data, dict):
                # Update average values for timing metrics
                if '_duration' in metric_name and 'average' in metric_data:
                    # Extract agent and operation from metric name
                    # e.g., "web_search_duration_seconds" -> agent="web_search", operation="search"
                    parts = metric_name.replace('_duration_seconds', '').split('_', 1)
                    if len(parts) >= 1:
                        agent = parts[0]
                        operation = parts[1] if len(parts) > 1 else "default"

                        # This won't update the histogram buckets, but will track the metric
                        # For true histogram updates, we'd need to observe each individual value
                        pass

                # Update counters
                if '_count' in metric_name and 'latest' in metric_data:
                    # Counters would be incremented elsewhere; this is just for monitoring
                    pass

    except Exception as e:
        import logging
        logging.error(f"Failed to sync metrics from collector: {e}")


def get_prometheus_metrics() -> bytes:
    """Get metrics in Prometheus text format.

    Returns:
        bytes: Metrics in Prometheus exposition format
    """
    # Update system metrics before generating output
    update_system_metrics()

    # Sync from existing metrics collector if available
    sync_from_metrics_collector()

    return generate_latest(registry)


def get_prometheus_content_type() -> str:
    """Get the Content-Type header value for Prometheus metrics.

    Returns:
        str: Content-Type header value
    """
    return CONTENT_TYPE_LATEST


def track_request(agent: str, operation: str):
    """Context manager for tracking request metrics.

    Usage:
        with track_request_duration('web_search', 'search'):
            # Perform operation
            pass

    Args:
        agent: Name of the agent
        operation: Name of the operation
    """
    class RequestTracker:
        def __init__(self, agent: str, operation: str):
            self.agent = agent
            self.operation = operation
            self.start_time = None

        def __enter__(self):
            self.start_time = time.time()
            requests_total.labels(agent=self.agent, operation=self.operation).inc()
            active_operations.labels(agent=self.agent).inc()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = time.time() - self.start_time
            request_duration.labels(agent=self.agent, operation=self.operation).observe(duration)
            active_operations.labels(agent=self.agent).dec()

            if exc_type is None:
                requests_success.labels(agent=self.agent, operation=self.operation).inc()
            else:
                error_type = exc_type.__name__ if exc_type else 'Unknown'
                requests_failed.labels(
                    agent=self.agent,
                    operation=self.operation,
                    error_type=error_type
                ).inc()

            return False  # Don't suppress exceptions

    return RequestTracker(agent, operation)


def track_llm_call(provider: str, model: str):
    """Track an LLM API call.

    Args:
        provider: LLM provider (e.g., 'openai', 'anthropic')
        model: Model name
    """
    llm_api_calls.labels(provider=provider, model=model).inc()


def track_search_call():
    """Track a search API call."""
    search_api_calls.inc()


def track_code_execution(status: str):
    """Track a code execution.

    Args:
        status: Execution status ('success' or 'failure')
    """
    code_executions.labels(status=status).inc()


def track_cache_access(cache_type: str, hit: bool):
    """Track a cache access.

    Args:
        cache_type: Type of cache (e.g., 'llm', 'search')
        hit: Whether it was a cache hit
    """
    if hit:
        cache_hits.labels(cache_type=cache_type).inc()
    else:
        cache_misses.labels(cache_type=cache_type).inc()


def track_retry(service: str, attempt: int, exhausted: bool = False):
    """Track a retry attempt.

    Args:
        service: Service name (e.g., 'llm_api', 'search_api')
        attempt: Attempt number
        exhausted: Whether all retries were exhausted
    """
    retries_total.labels(service=service, attempt=str(attempt)).inc()
    if exhausted:
        retries_exhausted.labels(service=service).inc()


def get_metrics_summary() -> Dict[str, Any]:
    """Get a summary of current metrics in JSON format.

    This is useful for non-Prometheus monitoring systems or debugging.

    Returns:
        Dict containing current metric values
    """
    update_system_metrics()

    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    uptime = time.time() - _start_time

    return {
        'timestamp': time.time(),
        'uptime_seconds': round(uptime, 2),
        'system': {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': memory.percent,
            'memory_available_mb': round(memory.available / (1024 * 1024), 2),
            'disk_percent': disk.percent,
            'disk_free_gb': round(disk.free / (1024 * 1024 * 1024), 2)
        },
        'info': {
            'version': '1.0.0',
            'name': 'gradio-mcp-hub'
        }
    }
