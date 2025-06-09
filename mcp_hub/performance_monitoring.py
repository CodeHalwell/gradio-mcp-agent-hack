"""Performance monitoring and metrics collection for the MCP Hub."""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
from contextlib import contextmanager
from .logging_config import logger

@dataclass
class MetricPoint:
    """Single metric measurement."""
    timestamp: datetime
    metric_name: str
    value: float
    tags: Dict[str, str]

class MetricsCollector:
    """Collects and stores application metrics."""
    
    def __init__(self, max_points: int = 10000):
        """
        Initialize metrics collector.
        
        Args:
            max_points: Maximum number of metric points to store
        """
        self.max_points = max_points
        self.metrics = defaultdict(lambda: deque(maxlen=max_points))
        self.lock = threading.Lock()
        self.counters = defaultdict(int)
        self.timers = {}
        
        # Start system metrics collection thread
        self.system_thread = threading.Thread(target=self._collect_system_metrics, daemon=True)
        self.system_thread.start()
        logger.info("Metrics collector initialized")
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        if tags is None:
            tags = {}
        
        point = MetricPoint(
            timestamp=datetime.now(),
            metric_name=name,
            value=value,
            tags=tags
        )
        
        with self.lock:
            self.metrics[name].append(point)
    
    def increment_counter(self, name: str, amount: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self.lock:
            self.counters[name] += amount
        
        self.record_metric(f"{name}_count", self.counters[name], tags)
    
    @contextmanager
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_metric(f"{name}_duration_seconds", duration, tags)
    
    def get_metrics_summary(self, 
                          metric_name: Optional[str] = None, 
                          last_minutes: int = 5) -> Dict[str, Any]:
        """Get summary statistics for metrics."""
        cutoff_time = datetime.now() - timedelta(minutes=last_minutes)
        
        with self.lock:
            if metric_name:
                metrics_to_analyze = {metric_name: self.metrics[metric_name]}
            else:
                metrics_to_analyze = dict(self.metrics)
        
        summary = {}
        
        for name, points in metrics_to_analyze.items():
            recent_points = [p for p in points if p.timestamp >= cutoff_time]
            
            if not recent_points:
                continue
            
            values = [p.value for p in recent_points]
            summary[name] = {
                "count": len(values),
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "latest": values[-1] if values else 0,
                "last_updated": recent_points[-1].timestamp.isoformat() if recent_points else None
            }
        
        return summary
    
    def _collect_system_metrics(self):
        """Background thread to collect system metrics."""
        while True:
            try:
                # CPU and memory metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                self.record_metric("system_cpu_percent", cpu_percent)
                self.record_metric("system_memory_percent", memory.percent)
                self.record_metric("system_memory_available_mb", memory.available / 1024 / 1024)
                
                # Process-specific metrics
                process = psutil.Process()
                process_memory = process.memory_info()
                
                self.record_metric("process_memory_rss_mb", process_memory.rss / 1024 / 1024)
                self.record_metric("process_cpu_percent", process.cpu_percent())
                
                time.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                time.sleep(60)  # Wait longer if there's an error

class PerformanceProfiler:
    """Profile performance of agent operations."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.operation_stats = defaultdict(list)
    
    @contextmanager
    def profile_operation(self, operation_name: str, **tags):
        """Context manager to profile an operation."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            yield
            success = True
        except Exception as e:
            success = False
            logger.error(f"Operation {operation_name} failed: {e}")
            raise
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            duration = end_time - start_time
            memory_delta = (end_memory - start_memory) / 1024 / 1024  # MB
            
            # Record metrics
            operation_tags = {"operation": operation_name, "success": str(success), **tags}
            self.metrics.record_metric("operation_duration_seconds", duration, operation_tags)
            self.metrics.record_metric("operation_memory_delta_mb", memory_delta, operation_tags)
            
            # Update operation stats
            self.operation_stats[operation_name].append({
                "duration": duration,
                "memory_delta": memory_delta,
                "success": success,
                "timestamp": datetime.now()
            })
    
    def get_operation_summary(self, operation_name: str = None) -> Dict[str, Any]:
        """Get summary of operation performance."""
        if operation_name:
            operations_to_analyze = {operation_name: self.operation_stats[operation_name]}
        else:
            operations_to_analyze = dict(self.operation_stats)
        
        summary = {}
        
        for op_name, stats in operations_to_analyze.items():
            if not stats:
                continue
            
            durations = [s["duration"] for s in stats]
            memory_deltas = [s["memory_delta"] for s in stats]
            success_rate = sum(1 for s in stats if s["success"]) / len(stats)
            
            summary[op_name] = {
                "total_calls": len(stats),
                "success_rate": success_rate,
                "avg_duration_seconds": sum(durations) / len(durations),
                "avg_memory_delta_mb": sum(memory_deltas) / len(memory_deltas),
                "min_duration": min(durations),
                "max_duration": max(durations)
            }
        
        return summary

# Global instances
metrics_collector = MetricsCollector()
performance_profiler = PerformanceProfiler(metrics_collector)

# Convenience decorators
def track_performance(operation_name: str = None):
    """Decorator to automatically track function performance."""
    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"
        
        def wrapper(*args, **kwargs):
            with performance_profiler.profile_operation(operation_name):
                result = func(*args, **kwargs)
                metrics_collector.increment_counter(f"{operation_name}_calls")
                return result
        return wrapper
    return decorator

def track_api_call(service_name: str):
    """Decorator specifically for tracking API calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with performance_profiler.profile_operation("api_call", service=service_name):
                try:
                    result = func(*args, **kwargs)
                    metrics_collector.increment_counter("api_calls_success", tags={"service": service_name})
                    return result
                except Exception:
                    metrics_collector.increment_counter("api_calls_failed", tags={"service": service_name})
                    raise
        return wrapper
    return decorator
