# Prometheus Metrics Export

The MCP Hub provides comprehensive metrics export in Prometheus format for production monitoring, alerting, and observability.

## Overview

The Prometheus metrics integration provides:
- **System Metrics**: CPU, memory, disk usage
- **Application Metrics**: Request counts, durations, success/failure rates
- **API Metrics**: LLM API calls, search API calls, retry attempts
- **Cache Metrics**: Cache hits and misses
- **Code Execution Metrics**: Execution counts by status

## Installation

Install the required dependency:

```bash
pip install prometheus-client>=0.20.0
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

## Accessing Metrics

### Via Gradio UI

1. Navigate to the **Advanced Features** tab
2. Click the **Get Prometheus Metrics** button
3. View the metrics in Prometheus text format

### Via API Endpoint

Access metrics programmatically:

```python
import requests

response = requests.get('http://localhost:7860/api/get_prometheus_metrics_service')
metrics = response.text
print(metrics)
```

## Available Metrics

### Application Info

- `mcp_hub_app_info` - Application metadata (version, name)

### Request Metrics

- `mcp_hub_requests_total` - Total number of requests by agent and operation
  - Labels: `agent`, `operation`

- `mcp_hub_requests_success_total` - Successful requests
  - Labels: `agent`, `operation`

- `mcp_hub_requests_failed_total` - Failed requests
  - Labels: `agent`, `operation`, `error_type`

- `mcp_hub_request_duration_seconds` - Request duration histogram
  - Labels: `agent`, `operation`
  - Buckets: 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, +Inf

- `mcp_hub_active_operations` - Currently active operations
  - Labels: `agent`

### System Metrics

- `mcp_hub_cpu_usage_percent` - CPU usage percentage
- `mcp_hub_memory_usage_percent` - Memory usage percentage
- `mcp_hub_memory_available_bytes` - Available memory in bytes
- `mcp_hub_disk_usage_percent` - Disk usage percentage
- `mcp_hub_disk_free_bytes` - Free disk space in bytes
- `mcp_hub_uptime_seconds` - Application uptime in seconds

### API Metrics

- `mcp_hub_llm_api_calls_total` - LLM API calls
  - Labels: `provider`, `model`

- `mcp_hub_search_api_calls_total` - Search API calls

- `mcp_hub_code_executions_total` - Code execution attempts
  - Labels: `status` (success/failure)

### Cache Metrics

- `mcp_hub_cache_hits_total` - Cache hits
  - Labels: `cache_type`

- `mcp_hub_cache_misses_total` - Cache misses
  - Labels: `cache_type`

### Retry Metrics

- `mcp_hub_retries_total` - Retry attempts
  - Labels: `service`, `attempt`

- `mcp_hub_retries_exhausted_total` - Times retries were exhausted
  - Labels: `service`

## Integration with Prometheus

### Basic Prometheus Configuration

Add this to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'mcp-hub'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:7860']
    metrics_path: '/api/get_prometheus_metrics_service'
```

### Docker Compose Example

```yaml
version: '3.8'

services:
  mcp-hub:
    build: .
    ports:
      - "7860:7860"
    environment:
      - TAVILY_API_KEY=${TAVILY_API_KEY}
      - NEBIUS_API_KEY=${NEBIUS_API_KEY}

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
```

### Kubernetes Service Monitor

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: mcp-hub
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: mcp-hub
  endpoints:
    - port: http
      path: /api/get_prometheus_metrics_service
      interval: 15s
```

## Grafana Dashboards

### Example Queries

**Request Rate**:
```promql
rate(mcp_hub_requests_total[5m])
```

**Error Rate**:
```promql
rate(mcp_hub_requests_failed_total[5m]) / rate(mcp_hub_requests_total[5m])
```

**Average Request Duration**:
```promql
rate(mcp_hub_request_duration_seconds_sum[5m]) / rate(mcp_hub_request_duration_seconds_count[5m])
```

**Memory Usage**:
```promql
mcp_hub_memory_usage_percent
```

**95th Percentile Response Time**:
```promql
histogram_quantile(0.95, rate(mcp_hub_request_duration_seconds_bucket[5m]))
```

## Alerting Rules

### Example Prometheus Alerts

```yaml
groups:
  - name: mcp_hub_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(mcp_hub_requests_failed_total[5m]) / rate(mcp_hub_requests_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: HighMemoryUsage
        expr: mcp_hub_memory_usage_percent > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}%"

      - alert: SlowResponses
        expr: histogram_quantile(0.95, rate(mcp_hub_request_duration_seconds_bucket[5m])) > 30
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow response times"
          description: "95th percentile response time is {{ $value }}s"

      - alert: ServiceDown
        expr: up{job="mcp-hub"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "MCP Hub service is down"
          description: "Service has been down for more than 1 minute"
```

## Custom Tracking

### Track Custom Operations

```python
from mcp_hub.prometheus_metrics import track_request

def my_custom_function():
    with track_request('custom_agent', 'custom_operation'):
        # Your operation here
        pass
```

### Track LLM Calls

```python
from mcp_hub.prometheus_metrics import track_llm_call

track_llm_call('openai', 'gpt-4')
```

### Track Cache Access

```python
from mcp_hub.prometheus_metrics import track_cache_access

# Cache hit
track_cache_access('llm', hit=True)

# Cache miss
track_cache_access('llm', hit=False)
```

### Track Retries

```python
from mcp_hub.prometheus_metrics import track_retry

# Track retry attempt
track_retry('llm_api', attempt=1)

# Track exhausted retries
track_retry('llm_api', attempt=3, exhausted=True)
```

## Best Practices

1. **Scrape Interval**: Use 15-30 second intervals for production
2. **Retention**: Configure Prometheus retention based on your needs (default: 15 days)
3. **Alerting**: Set up alerts for:
   - High error rates (>10%)
   - High memory usage (>90%)
   - Slow response times (>30s at p95)
   - Service downtime
4. **Dashboard**: Create Grafana dashboards for:
   - Request rate and error rate
   - Response time percentiles
   - System resource usage
   - API call distribution

## Troubleshooting

### Metrics Not Available

If you see "Prometheus metrics not available":

```bash
pip install prometheus-client>=0.20.0
```

### Empty Metrics

Metrics are collected over time. Generate some traffic first:
1. Make some requests through the UI
2. Wait 15-30 seconds
3. Refresh the metrics endpoint

### High Memory Usage

If metrics collection causes high memory:
- Reduce scrape interval
- Reduce histogram bucket count
- Consider using recording rules in Prometheus

## Architecture

The metrics system consists of:

1. **prometheus_metrics.py**: Core metrics definitions and export
2. **Integration Points**: Decorators in agents and utilities
3. **Export Endpoint**: Gradio API endpoint for Prometheus scraping
4. **Bridge**: Integration with existing metrics_collector

## Further Reading

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Dashboards](https://grafana.com/docs/grafana/latest/dashboards/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [OpenMetrics Specification](https://openmetrics.io/)
