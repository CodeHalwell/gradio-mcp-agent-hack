"""Simple health and metrics endpoints for production monitoring.

These endpoints can be used by load balancers, monitoring systems,
and orchestration platforms like Kubernetes for health checks.
"""

import time
import psutil
from typing import Dict, Any


_start_time = time.time()


def get_simple_health() -> Dict[str, Any]:
    """
    Get simple health status for load balancer health checks.

    Returns minimal information for fast health checking.

    Returns:
        Dict with status and uptime
    """
    try:
        uptime = time.time() - _start_time
        return {
            "status": "healthy",
            "uptime_seconds": round(uptime, 2),
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }


def get_detailed_health() -> Dict[str, Any]:
    """
    Get detailed health status including system metrics.

    Returns comprehensive health information for monitoring dashboards.

    Returns:
        Dict with detailed system and application health
    """
    try:
        uptime = time.time() - _start_time

        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime_seconds": round(uptime, 2),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": round(memory.available / (1024 * 1024), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024 * 1024 * 1024), 2)
            }
        }

        # Determine overall health status
        if cpu_percent > 90:
            health_status["status"] = "degraded"
            health_status["warnings"] = ["High CPU usage"]
        if memory.percent > 90:
            health_status["status"] = "degraded"
            health_status["warnings"] = health_status.get("warnings", []) + ["High memory usage"]
        if disk.percent > 90:
            health_status["status"] = "degraded"
            health_status["warnings"] = health_status.get("warnings", []) + ["Low disk space"]

        return health_status

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }


def get_readiness() -> Dict[str, Any]:
    """
    Get readiness status for Kubernetes readiness probes.

    Checks if the application is ready to receive traffic.

    Returns:
        Dict with readiness status
    """
    try:
        # Check if system has enough resources
        memory = psutil.virtual_memory()

        if memory.percent > 95:
            return {
                "ready": False,
                "reason": "Insufficient memory",
                "memory_percent": memory.percent
            }

        return {
            "ready": True,
            "timestamp": time.time()
        }

    except Exception as e:
        return {
            "ready": False,
            "error": str(e),
            "timestamp": time.time()
        }


def get_metrics() -> Dict[str, Any]:
    """
    Get basic metrics in a simple format.

    Returns metrics that can be scraped by monitoring systems.

    Returns:
        Dict with application metrics
    """
    try:
        uptime = time.time() - _start_time
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        return {
            "uptime_seconds": round(uptime, 2),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_mb": round(memory.used / (1024 * 1024), 2),
            "timestamp": time.time()
        }

    except Exception as e:
        return {
            "error": str(e),
            "timestamp": time.time()
        }
