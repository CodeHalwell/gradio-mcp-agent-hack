## Advanced Performance Monitoring

Comprehensive guide to the MCP Hub's advanced performance monitoring and profiling capabilities.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Request Tracing](#request-tracing)
- [Slow Query Detection](#slow-query-detection)
- [Performance Bottlenecks](#performance-bottlenecks)
- [Memory Profiling](#memory-profiling)
- [API Endpoints](#api-endpoints)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The Advanced Monitoring system provides deep insights into application performance, including:
- **Distributed request tracing** with nested spans
- **Automatic slow query detection** with configurable thresholds
- **Performance bottleneck identification** with actionable recommendations
- **Real-time memory profiling** with historical data
- **Comprehensive metrics aggregation** by operation type

This goes beyond basic metrics to provide actionable intelligence for performance optimization.

## Features

### 1. Request Tracing

Every request is traced with a unique trace ID, capturing:
- Start and end times
- Total duration
- Success/failure status
- Custom metadata
- Nested spans for sub-operations

```python
{
  "trace_id": "550e8400-e29b-41d4-a716-446655440000",
  "operation": "orchestrate",
  "start_time": 1735734000.123,
  "end_time": 1735734015.456,
  "duration": 15.333,
  "status": "success",
  "metadata": {"user_request": "How do I use async/await?"},
  "spans": [
    {"name": "enhance_question", "duration": 2.1},
    {"name": "web_search", "duration": 1.8},
    {"name": "generate_code", "duration": 8.5},
    {"name": "run_code", "duration": 2.3}
  ]
}
```

### 2. Slow Query Detection

Automatically detects operations exceeding threshold:
- **Default threshold**: 5 seconds (configurable)
- **Automatic logging**: Slow queries are logged as warnings
- **Historical tracking**: Last 100 slow queries retained
- **Detailed context**: Includes operation metadata

### 3. Bottleneck Detection

Analyzes patterns to identify:
- **Slow operations**: Average duration exceeds threshold
- **High error rates**: Error rate > 10%
- **Memory issues**: Memory usage > 80%

Each bottleneck includes actionable recommendations.

### 4. Memory Profiling

Continuous memory tracking every 30 seconds:
- **RSS (Resident Set Size)**: Physical memory used
- **VMS (Virtual Memory Size)**: Total memory allocated
- **Percentage**: System memory usage
- **Available memory**: Free memory on system

Historical data allows trend analysis and leak detection.

### 5. Operation Statistics

Aggregated stats for each operation:
- **Count**: Total invocations
- **Average duration**: Mean execution time
- **Min/Max duration**: Performance bounds
- **Error count**: Total failures
- **Error rate**: Percentage of failures

## Installation

### Basic Requirements

```bash
pip install psutil>=5.9.0
```

### Optional: Memory Profiler

For advanced memory profiling:

```bash
pip install memory-profiler
```

### Verify Installation

```python
from mcp_hub.advanced_monitoring import advanced_monitor
print("Advanced monitoring available!")
```

## Quick Start

### Via Gradio UI

1. Start the MCP Hub:
   ```bash
   python app.py
   ```

2. Navigate to the **"Advanced Monitoring"** tab

3. Click buttons to access different views:
   - **Get Performance Report**: Comprehensive overview
   - **Get Request Traces**: Recent and active traces
   - **Get Slow Queries**: Operations exceeding threshold
   - **Detect Bottlenecks**: Performance issues and recommendations

### Via API

```python
import requests

# Get performance report
response = requests.post(
    'http://localhost:7860/api/get_advanced_performance_report_service',
    json={"data": []}
)
report = response.json()['data'][0]
print(f"Total traces: {report['total_traces_count']}")

# Get request traces
response = requests.post(
    'http://localhost:7860/api/get_request_traces_service',
    json={"data": []}
)
traces = response.json()['data'][0]
print(f"Recent traces: {len(traces['recent_traces'])}")

# Detect bottlenecks
response = requests.post(
    'http://localhost:7860/api/get_performance_bottlenecks_service',
    json={"data": []}
)
bottlenecks = response.json()['data'][0]
if bottlenecks['has_issues']:
    print(f"Found {bottlenecks['count']} bottlenecks!")
```

## Request Tracing

### Automatic Tracing

All MCP Hub operations are automatically traced. No configuration needed!

### Manual Tracing in Custom Code

#### Using Context Manager

```python
from mcp_hub.advanced_monitoring import advanced_monitor

def my_custom_operation():
    with advanced_monitor.trace_request('custom_op', {'key': 'value'}):
        # Your operation here
        result = perform_work()
        return result
```

#### Using Decorator

```python
from mcp_hub.advanced_monitoring import trace_operation

@trace_operation("my_function")
def my_function(arg1, arg2):
    # Function body
    return result

# Async functions work too!
@trace_operation("my_async_function")
async def my_async_function():
    result = await some_async_work()
    return result
```

### Tracing Spans (Sub-Operations)

Track nested operations within a request:

```python
with advanced_monitor.trace_request('orchestrate') as trace:
    # Step 1: Search
    with advanced_monitor.trace_span(trace, 'web_search', {'query': 'python'}):
        search_results = web_search_agent.search(query)

    # Step 2: Process
    with advanced_monitor.trace_span(trace, 'llm_process'):
        processed = llm_processor.process(search_results)

    # Step 3: Generate code
    with advanced_monitor.trace_span(trace, 'code_generation'):
        code = code_generator.generate(processed)
```

### Accessing Trace Data

```python
# Get recent traces
recent = advanced_monitor.get_recent_traces(limit=10)
for trace in recent:
    print(f"{trace['operation']}: {trace['duration']:.2f}s")
    for span in trace['spans']:
        print(f"  - {span['name']}: {span['duration']:.2f}s")

# Get active (running) traces
active = advanced_monitor.get_active_traces()
print(f"Currently running: {len(active)} operations")
```

## Slow Query Detection

### Configuration

Set threshold when initializing (default: 5 seconds):

```python
from mcp_hub.advanced_monitoring import AdvancedMonitor

# Custom threshold
monitor = AdvancedMonitor(slow_query_threshold=3.0)
```

### Automatic Detection

When an operation exceeds the threshold:

1. **Automatic logging**: Warning logged with operation name and duration
2. **Slow query record created**: Stored in memory
3. **Accessible via API**: Retrievable for analysis

Example log output:
```
WARNING Slow query detected: orchestrate took 8.45s
```

### Retrieving Slow Queries

```python
# Via API
slow_queries = advanced_monitor.get_slow_queries(limit=10)
for sq in slow_queries:
    print(f"Operation: {sq['operation']}")
    print(f"Duration: {sq['duration']:.2f}s")
    print(f"Timestamp: {sq['timestamp']}")
    print(f"Trace ID: {sq['trace_id']}")
    print(f"Metadata: {sq['metadata']}")
```

### Example Output

```json
{
  "slow_queries": [
    {
      "operation": "orchestrate",
      "duration": 12.5,
      "timestamp": "2025-01-06T10:30:45",
      "trace_id": "550e8400-e29b-41d4-a716-446655440000",
      "metadata": {"user_request": "Complex analysis task"}
    },
    {
      "operation": "web_search",
      "duration": 8.2,
      "timestamp": "2025-01-06T10:28:15",
      "trace_id": "660f9511-f30c-52e5-b827-557766551111",
      "metadata": {"query": "Very specific technical query"}
    }
  ],
  "threshold_seconds": 5.0
}
```

## Performance Bottlenecks

### Automatic Detection

The system continuously analyzes performance data to detect:

#### 1. Slow Operations

**Criteria**: Average duration > threshold

```json
{
  "type": "slow_operation",
  "operation": "code_generation",
  "average_duration": 8.5,
  "count": 25,
  "recommendation": "Consider optimizing code_generation"
}
```

**Actions**:
- Review operation implementation
- Check for unnecessary work
- Consider caching results
- Optimize LLM prompts

#### 2. High Error Rates

**Criteria**: Error rate > 10%

```json
{
  "type": "high_error_rate",
  "operation": "web_search",
  "error_rate": 0.15,
  "errors": 15,
  "recommendation": "Investigate errors in web_search"
}
```

**Actions**:
- Check logs for error patterns
- Verify API credentials
- Implement better error handling
- Add retry logic

#### 3. High Memory Usage

**Criteria**: Memory usage > 80%

```json
{
  "type": "high_memory_usage",
  "current_percent": 85.2,
  "current_rss_mb": 3456.7,
  "recommendation": "Memory usage is high, consider optimization"
}
```

**Actions**:
- Review memory-intensive operations
- Check for memory leaks
- Implement result streaming
- Clear caches periodically

### Retrieving Bottlenecks

```python
bottlenecks = advanced_monitor.detect_bottlenecks()

if bottlenecks:
    print(f"⚠️  Found {len(bottlenecks)} performance issues:\n")
    for b in bottlenecks:
        print(f"Type: {b['type']}")
        print(f"Recommendation: {b['recommendation']}\n")
else:
    print("✅ No bottlenecks detected!")
```

## Memory Profiling

### Automatic Tracking

Memory is tracked every 30 seconds in the background:
- **Non-blocking**: Runs in separate thread
- **Minimal overhead**: Lightweight monitoring
- **Historical data**: Last 1000 snapshots (≈8 hours at 30s intervals)

### Accessing Memory Stats

```python
# Get memory stats for last 5 minutes
memory_stats = advanced_monitor.get_memory_stats(minutes=5)

print(f"Current RSS: {memory_stats['current_rss_mb']:.1f} MB")
print(f"Average RSS: {memory_stats['average_rss_mb']:.1f} MB")
print(f"Max RSS: {memory_stats['max_rss_mb']:.1f} MB")
print(f"Memory %: {memory_stats['current_percent']:.1f}%")
```

### Example Output

```json
{
  "current_rss_mb": 456.7,
  "current_percent": 5.8,
  "average_rss_mb": 423.4,
  "average_percent": 5.4,
  "max_rss_mb": 512.3,
  "max_percent": 6.5,
  "min_rss_mb": 389.2,
  "min_percent": 4.9,
  "snapshots_count": 10,
  "time_window_minutes": 5
}
```

### Memory Leak Detection

Monitor memory trends over time:

```python
import time

# Baseline
baseline = advanced_monitor.get_memory_stats(minutes=1)
print(f"Baseline: {baseline['current_rss_mb']:.1f} MB")

# Perform operations
for i in range(100):
    # ... your operations ...
    pass

# Check after operations
time.sleep(60)  # Wait for monitoring to update
after = advanced_monitor.get_memory_stats(minutes=1)
print(f"After: {after['current_rss_mb']:.1f} MB")

growth = after['current_rss_mb'] - baseline['current_rss_mb']
if growth > 100:  # More than 100 MB growth
    print("⚠️  Potential memory leak detected!")
```

## API Endpoints

### 1. Get Performance Report

**Endpoint**: `/api/get_advanced_performance_report_service`

**Description**: Comprehensive performance overview

**Request**:
```json
{
  "data": []
}
```

**Response**:
```json
{
  "data": [
    {
      "timestamp": "2025-01-06T10:30:00",
      "operation_stats": {
        "web_search": {
          "count": 150,
          "average_duration": 1.8,
          "min_duration": 0.5,
          "max_duration": 8.2,
          "errors": 5,
          "error_rate": 0.033
        }
      },
      "memory_stats": {
        "current_rss_mb": 456.7,
        "average_percent": 5.4
      },
      "slow_queries": [...],
      "active_traces_count": 2,
      "total_traces_count": 1000,
      "recent_traces": [...]
    }
  ]
}
```

### 2. Get Request Traces

**Endpoint**: `/api/get_request_traces_service`

**Response**:
```json
{
  "data": [
    {
      "recent_traces": [...],
      "active_traces": [...]
    }
  ]
}
```

### 3. Get Slow Queries

**Endpoint**: `/api/get_slow_queries_service`

**Response**:
```json
{
  "data": [
    {
      "slow_queries": [...],
      "threshold_seconds": 5.0
    }
  ]
}
```

### 4. Detect Bottlenecks

**Endpoint**: `/api/get_performance_bottlenecks_service`

**Response**:
```json
{
  "data": [
    {
      "bottlenecks": [...],
      "count": 2,
      "has_issues": true
    }
  ]
}
```

## Best Practices

### 1. Set Appropriate Thresholds

Adjust slow query threshold based on your use case:

```python
# Development/testing: Lower threshold for strict monitoring
monitor = AdvancedMonitor(slow_query_threshold=2.0)

# Production: Higher threshold for actual slow queries
monitor = AdvancedMonitor(slow_query_threshold=10.0)
```

### 2. Use Spans for Complex Operations

Break down complex operations into spans:

```python
with advanced_monitor.trace_request('complex_workflow') as trace:
    # Phase 1
    with advanced_monitor.trace_span(trace, 'data_collection'):
        data = collect_data()

    # Phase 2
    with advanced_monitor.trace_span(trace, 'data_processing'):
        processed = process_data(data)

    # Phase 3
    with advanced_monitor.trace_span(trace, 'result_generation'):
        result = generate_result(processed)
```

This reveals which phase is the bottleneck!

### 3. Regular Bottleneck Checks

Schedule periodic bottleneck detection:

```python
import schedule

def check_bottlenecks():
    bottlenecks = advanced_monitor.detect_bottlenecks()
    if bottlenecks:
        # Send alert, log, or take action
        alert_team(bottlenecks)

# Check every hour
schedule.every().hour.do(check_bottlenecks)
```

### 4. Monitor Memory Trends

Set up alerts for memory growth:

```python
def check_memory():
    stats = advanced_monitor.get_memory_stats(minutes=30)
    if stats['current_percent'] > 80:
        send_alert(f"High memory usage: {stats['current_percent']:.1f}%")

schedule.every(10).minutes.do(check_memory)
```

### 5. Reset Stats Periodically

For long-running applications:

```python
# Reset weekly to prevent unbounded growth
schedule.every().week.do(advanced_monitor.reset_stats)
```

### 6. Include Meaningful Metadata

Add context to traces:

```python
with advanced_monitor.trace_request(
    'user_request',
    metadata={
        'user_id': user.id,
        'request_type': 'code_generation',
        'complexity': 'high'
    }
):
    # Operation
    pass
```

## Troubleshooting

### Issue: No Traces Appearing

**Symptoms**: `get_recent_traces()` returns empty list

**Solutions**:
1. Verify operations are being traced:
   ```python
   with advanced_monitor.trace_request('test'):
       pass
   traces = advanced_monitor.get_recent_traces()
   assert len(traces) > 0
   ```

2. Check max_traces limit hasn't been reached with all old traces:
   ```python
   advanced_monitor.reset_stats()  # Clear old traces
   ```

### Issue: Memory Tracking Not Working

**Symptoms**: `get_memory_stats()` returns error

**Solutions**:
1. Verify psutil is installed:
   ```bash
   pip install psutil>=5.9.0
   ```

2. Check memory tracking is enabled:
   ```python
   monitor = AdvancedMonitor(memory_tracking_enabled=True)
   ```

3. Wait for snapshots to accumulate:
   ```python
   import time
   time.sleep(60)  # Wait for background thread
   stats = monitor.get_memory_stats()
   ```

### Issue: Slow Queries Not Detected

**Symptoms**: No slow queries despite slow operations

**Solutions**:
1. Check threshold configuration:
   ```python
   print(f"Threshold: {advanced_monitor.slow_query_threshold}s")
   ```

2. Lower threshold for testing:
   ```python
   advanced_monitor.slow_query_threshold = 1.0
   ```

3. Verify operation duration:
   ```python
   traces = advanced_monitor.get_recent_traces(limit=5)
   for t in traces:
       print(f"{t['operation']}: {t['duration']:.2f}s")
   ```

### Issue: High Memory Usage

**Symptoms**: Monitor itself using too much memory

**Solutions**:
1. Reduce max_traces:
   ```python
   monitor = AdvancedMonitor(max_traces=500)
   ```

2. Reset stats more frequently:
   ```python
   schedule.every().day.do(monitor.reset_stats)
   ```

3. Disable memory tracking if not needed:
   ```python
   monitor = AdvancedMonitor(memory_tracking_enabled=False)
   ```

## Integration with Prometheus

Export advanced monitoring metrics to Prometheus:

```python
from mcp_hub.prometheus_metrics import (
    track_request,
    track_llm_call,
    track_cache_access
)

# Automatic integration
with advanced_monitor.trace_request('operation') as trace:
    with track_request('agent', 'operation'):
        # Both systems track this operation
        pass
```

The Prometheus metrics endpoint includes aggregated data from advanced monitoring.

## Further Reading

- [Prometheus Metrics Guide](PROMETHEUS.md)
- [Redis Caching Documentation](REDIS_CACHE.md)
- [API Documentation](API_DOCUMENTATION.md)
- [MCP Protocol Schema](mcp_schema.json)

## Support

For issues or questions:
- Check [Troubleshooting](#troubleshooting) section
- Review [Best Practices](#best-practices)
- Open an issue on GitHub
