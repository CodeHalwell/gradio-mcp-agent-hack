"""System health monitoring and status dashboard functionality."""

import time
import psutil
from datetime import datetime
from typing import Dict, Any
from .config import api_config
from .logging_config import logger
from .reliability_utils import health_monitor
from .performance_monitoring import metrics_collector

class SystemHealthChecker:
    """Comprehensive system health checking."""
    
    def __init__(self):
        self.last_check = None
        self.health_status = {}
    
    def check_api_connectivity(self) -> Dict[str, Any]:
        """Check connectivity to external APIs."""
        results = {}
        
        # Check Nebius API
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=api_config.nebius_api_key,
                base_url=api_config.nebius_base_url
            )
            
            start_time = time.time()
            # Make a minimal test call
            response = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            response_time = time.time() - start_time
            
            results["nebius"] = {
                "status": "healthy",
                "response_time_ms": response_time * 1000,
                "last_checked": datetime.now().isoformat()
            }
            
        except Exception as e:
            results["nebius"] = {
                "status": "unhealthy",
                "error": str(e),
                "last_checked": datetime.now().isoformat()
            }
        
        # Check Tavily API
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_config.tavily_api_key)
            
            start_time = time.time()
            # Make a minimal test search
            response = client.search(query="test", max_results=1)
            response_time = time.time() - start_time
            
            results["tavily"] = {
                "status": "healthy",
                "response_time_ms": response_time * 1000,
                "last_checked": datetime.now().isoformat()
            }
            
        except Exception as e:
            results["tavily"] = {
                "status": "unhealthy", 
                "error": str(e),
                "last_checked": datetime.now().isoformat()
            }
        
        return results
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "percent_used": memory.percent
                },
                "disk": {
                    "total_gb": disk.total / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "percent_used": (disk.used / disk.total) * 100
                },
                "process": {
                    "memory_mb": process_memory.rss / (1024**2),
                    "cpu_percent": process.cpu_percent()
                },
                "status": "healthy",
                "last_checked": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_checked": datetime.now().isoformat()
            }
    
    def check_cache_health(self) -> Dict[str, Any]:
        """Check cache system health."""
        try:
            from cache_utils import cache_manager
            
            # Count cache files
            cache_files = list(cache_manager.cache_dir.glob("*.cache"))
            
            # Calculate cache directory size
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                "cache_files_count": len(cache_files),
                "total_size_mb": total_size / (1024**2),
                "cache_directory": str(cache_manager.cache_dir),
                "status": "healthy",
                "last_checked": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_checked": datetime.now().isoformat()
            }
    
    def get_comprehensive_health_report(self) -> Dict[str, Any]:
        """Get a comprehensive health report of the entire system."""
        logger.info("Generating comprehensive health report")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy"  # Will be updated based on checks
        }
        
        # Check API connectivity
        api_health = self.check_api_connectivity()
        report["api_connectivity"] = api_health
        
        # Check system resources
        system_health = self.check_system_resources()
        report["system_resources"] = system_health
        
        # Check cache health
        cache_health = self.check_cache_health()
        report["cache_system"] = cache_health
        
        # Get API health stats from monitor
        try:
            nebius_stats = health_monitor.get_health_stats("nebius")
            tavily_stats = health_monitor.get_health_stats("tavily")
            
            report["api_performance"] = {
                "nebius": nebius_stats,
                "tavily": tavily_stats
            }
        except Exception as e:
            report["api_performance"] = {"error": str(e)}
        
        # Get performance metrics
        try:
            performance_summary = metrics_collector.get_metrics_summary()
            report["performance_metrics"] = performance_summary
        except Exception as e:
            report["performance_metrics"] = {"error": str(e)}
        
        # Determine overall status
        unhealthy_components = []
        
        for service, status in api_health.items():
            if status.get("status") == "unhealthy":
                unhealthy_components.append(f"API:{service}")
        
        if system_health.get("status") == "unhealthy":
            unhealthy_components.append("system_resources")
        
        if cache_health.get("status") == "unhealthy":
            unhealthy_components.append("cache_system")
        
        if unhealthy_components:
            report["overall_status"] = "degraded"
            report["unhealthy_components"] = unhealthy_components
        
        self.last_check = datetime.now()
        self.health_status = report
        
        logger.info(f"Health report generated: {report['overall_status']}")
        return report

# Global health checker instance
health_checker = SystemHealthChecker()

def create_health_dashboard() -> str:
    """Create a formatted health dashboard for display."""
    report = health_checker.get_comprehensive_health_report()
    
    dashboard = f"""
# 🏥 System Health Dashboard
**Last Updated:** {report['timestamp']}  
**Overall Status:** {'🟢' if report['overall_status'] == 'healthy' else '🟡' if report['overall_status'] == 'degraded' else '🔴'} {report['overall_status'].upper()}

## 🌐 API Connectivity
"""
    
    for service, status in report.get("api_connectivity", {}).items():
        status_icon = "🟢" if status.get("status") == "healthy" else "🔴"
        response_time = status.get("response_time_ms", 0)
        dashboard += f"- **{service.title()}:** {status_icon} {status.get('status', 'unknown')} ({response_time:.1f}ms)\n"
    
    dashboard += "\n## 💻 System Resources\n"
    sys_resources = report.get("system_resources", {})
    if "memory" in sys_resources:
        memory = sys_resources["memory"]
        dashboard += f"- **Memory:** {memory['percent_used']:.1f}% used ({memory['available_gb']:.1f}GB available)\n"
    
    if "cpu_percent" in sys_resources:
        dashboard += f"- **CPU:** {sys_resources['cpu_percent']:.1f}% usage\n"
    
    if "process" in sys_resources:
        process = sys_resources["process"]
        dashboard += f"- **Process Memory:** {process['memory_mb']:.1f}MB\n"
    
    dashboard += "\n## 📊 Performance Metrics\n"
    perf_metrics = report.get("performance_metrics", {})
    if perf_metrics and not perf_metrics.get("error"):
        for metric_name, metric_data in perf_metrics.items():
            if isinstance(metric_data, dict) and "average" in metric_data:
                dashboard += f"- **{metric_name}:** Avg: {metric_data['average']:.3f}, Count: {metric_data['count']}\n"
    
    dashboard += "\n## 🔧 Cache System\n"
    cache_info = report.get("cache_system", {})
    if cache_info.get("status") == "healthy":
        dashboard += f"- **Cache Files:** {cache_info.get('cache_files_count', 0)} files\n"
        dashboard += f"- **Cache Size:** {cache_info.get('total_size_mb', 0):.1f}MB\n"
    
    if report.get("unhealthy_components"):
        dashboard += "\n## ⚠️ Issues Detected\n"
        for component in report["unhealthy_components"]:
            dashboard += f"- {component}\n"
    
    return dashboard
