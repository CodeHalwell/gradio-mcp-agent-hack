## WebSocket Support for Real-Time Streaming

Complete guide to using WebSocket for real-time progress streaming during long-running operations in the MCP Hub.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Server Configuration](#server-configuration)
- [Client Examples](#client-examples)
- [Event Types](#event-types)
- [API Reference](#api-reference)
- [Integration with Agents](#integration-with-agents)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

WebSocket support enables real-time streaming of progress updates for long-running operations. Instead of polling or waiting for completion, clients receive instant updates as operations progress through their various stages.

**Use Cases**:
- Code generation progress
- Web search updates
- Multi-step orchestration workflows
- Real-time execution monitoring
- Live logging and status updates

## Features

### Real-Time Progress Streaming
- **Progress Updates**: 0-100% completion tracking
- **Status Changes**: Running, processing, completed, error
- **Live Logging**: Info, warning, error messages
- **Result Delivery**: Final results pushed to client
- **Error Notifications**: Instant error alerts

### Operation Management
- **Unique Operation IDs**: Track individual operations
- **Subscription Model**: Subscribe to specific operations
- **Multiple Clients**: Multiple clients can monitor same operation
- **Operation History**: Access historical operation data

### Server Features
- **Background Thread**: Runs alongside Gradio without blocking
- **Graceful Shutdown**: Clean connection termination
- **Connection Management**: Automatic client registration/cleanup
- **Statistics**: Real-time server statistics

## Installation

### Requirements

```bash
pip install websockets>=12.0
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

### Verify Installation

```python
from mcp_hub.websocket_server import WEBSOCKETS_AVAILABLE
print(f"WebSocket available: {WEBSOCKETS_AVAILABLE}")
```

## Quick Start

### 1. Start the MCP Hub

The WebSocket server starts automatically with the MCP Hub:

```bash
python app.py
```

Output:
```
INFO WebSocket manager initialized
INFO WebSocket server started on port 8765
INFO Starting WebSocket server on 0.0.0.0:8765
INFO WebSocket server started on ws://0.0.0.0:8765
```

### 2. Connect a Client

**Python Example**:

```python
import asyncio
import websockets
import json

async def connect():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        # Receive welcome message
        welcome = await websocket.recv()
        print(f"Connected: {welcome}")

        # Subscribe to an operation
        await websocket.send(json.dumps({
            "command": "subscribe",
            "operation_id": "your-operation-id"
        }))

        # Listen for updates
        async for message in websocket:
            event = json.loads(message)
            print(f"Event: {event['event_type']} - {event['data']}")

asyncio.run(connect())
```

**JavaScript Example**:

```javascript
const ws = new WebSocket('ws://localhost:8765');

ws.onopen = () => {
    console.log('Connected!');

    // Subscribe to operation
    ws.send(JSON.stringify({
        command: 'subscribe',
        operation_id: 'your-operation-id'
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Event:', data.event_type, data.data);
};
```

### 3. Use the HTML Client

Open `examples/websocket_client_javascript.html` in your browser for a full-featured WebSocket client with:
- Visual progress bars
- Event logging
- Server statistics
- Connection management

## Server Configuration

### Environment Variables

Configure WebSocket server via environment variables:

```bash
# Enable/disable WebSocket server
ENABLE_WEBSOCKET=true  # or false

# WebSocket server port
WEBSOCKET_PORT=8765
```

### Configuration in Code

```python
# Disable WebSocket
import os
os.environ["ENABLE_WEBSOCKET"] = "false"

# Custom port
os.environ["WEBSOCKET_PORT"] = "9000"

# Then start app
python app.py
```

### Manual Server Control

```python
from mcp_hub.websocket_launcher import (
    start_websocket_background,
    stop_websocket_background,
    is_websocket_running
)

# Start manually
start_websocket_background(host="0.0.0.0", port=8765)

# Check status
if is_websocket_running():
    print("WebSocket server is running")

# Stop manually
stop_websocket_background()
```

## Client Examples

### Python Client

Full example in `examples/websocket_client_python.py`:

```bash
# Stream specific operation
python examples/websocket_client_python.py <operation-id>

# Monitor all operations
python examples/websocket_client_python.py monitor

# Get server stats
python examples/websocket_client_python.py stats

# Interactive mode
python examples/websocket_client_python.py interactive
```

**Simple Monitoring**:

```python
import asyncio
import websockets
import json

async def monitor_operation(operation_id):
    uri = "ws://localhost:8765"

    async with websockets.connect(uri) as websocket:
        # Receive welcome
        await websocket.recv()

        # Subscribe
        await websocket.send(json.dumps({
            "command": "subscribe",
            "operation_id": operation_id
        }))

        print(f"Monitoring operation: {operation_id}\n")

        async for message in websocket:
            event = json.loads(message)

            if event['event_type'] == 'progress':
                progress = event['data']['progress'] * 100
                message = event['data']['message']
                print(f"[{progress:.0f}%] {message}")

            elif event['event_type'] == 'completed':
                duration = event['data']['duration']
                print(f"\n✅ Completed in {duration:.2f}s")
                break

            elif event['event_type'] == 'error':
                error = event['data']['error']
                print(f"\n❌ Error: {error}")
                break

asyncio.run(monitor_operation("your-operation-id"))
```

### JavaScript/Browser Client

Open `examples/websocket_client_javascript.html` in a browser or use the WebSocket API directly:

```javascript
class MCPWebSocketClient {
    constructor(url = 'ws://localhost:8765') {
        this.url = url;
        this.ws = null;
        this.listeners = {};
    }

    connect() {
        return new Promise((resolve, reject) => {
            this.ws = new WebSocket(this.url);

            this.ws.onopen = () => {
                console.log('Connected to MCP Hub');
                resolve();
            };

            this.ws.onerror = (error) => {
                reject(error);
            };

            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleEvent(data);
            };
        });
    }

    subscribe(operationId) {
        this.ws.send(JSON.stringify({
            command: 'subscribe',
            operation_id: operationId
        }));
    }

    on(eventType, callback) {
        if (!this.listeners[eventType]) {
            this.listeners[eventType] = [];
        }
        this.listeners[eventType].push(callback);
    }

    handleEvent(data) {
        const eventType = data.event_type;
        if (this.listeners[eventType]) {
            this.listeners[eventType].forEach(callback => {
                callback(data);
            });
        }
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
    }
}

// Usage
const client = new MCPWebSocketClient();

await client.connect();

client.on('progress', (event) => {
    console.log('Progress:', event.data.progress * 100 + '%');
});

client.on('completed', (event) => {
    console.log('Completed!');
});

client.subscribe('operation-id');
```

### curl (Testing)

WebSocket doesn't work with curl directly, but you can test the HTTP status endpoint:

```bash
curl -X POST http://localhost:7860/api/get_websocket_status_service \
  -H "Content-Type: application/json" \
  -d '{"data": []}'
```

## Event Types

### 1. Connected

Sent when client first connects.

```json
{
  "event_type": "connected",
  "operation_id": "system",
  "timestamp": 1735734000.123,
  "data": {
    "message": "Connected to MCP Hub WebSocket server",
    "client_id": 12345,
    "timestamp": "2025-01-06T10:30:00"
  }
}
```

### 2. Progress

Progress update during operation.

```json
{
  "event_type": "progress",
  "operation_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": 1735734005.456,
  "data": {
    "progress": 0.5,
    "message": "Generating code...",
    "details": {
      "step": "code_generation",
      "lines_generated": 42
    }
  }
}
```

### 3. Status

Status change notification.

```json
{
  "event_type": "status",
  "operation_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": 1735734010.789,
  "data": {
    "status": "processing",
    "message": "Processing search results",
    "details": {
      "results_found": 5
    }
  }
}
```

### 4. Log

Log message from operation.

```json
{
  "event_type": "log",
  "operation_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": 1735734012.345,
  "data": {
    "level": "info",
    "message": "Successfully validated code"
  }
}
```

### 5. Result

Final operation result.

```json
{
  "event_type": "result",
  "operation_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": 1735734015.678,
  "data": {
    "result": {
      "code": "def hello():\n    print('Hello, World!')",
      "validation_passed": true
    }
  }
}
```

### 6. Completed

Operation completed successfully.

```json
{
  "event_type": "completed",
  "operation_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": 1735734015.890,
  "data": {
    "status": "completed",
    "duration": 15.767
  }
}
```

### 7. Error

Operation failed with error.

```json
{
  "event_type": "error",
  "operation_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": 1735734010.123,
  "data": {
    "error": "Code validation failed",
    "error_type": "ValidationError"
  }
}
```

## API Reference

### Client Commands

Send commands to the WebSocket server:

#### Subscribe to Operation

```json
{
  "command": "subscribe",
  "operation_id": "your-operation-id"
}
```

**Response**:
```json
{
  "status": "subscribed",
  "operation_id": "your-operation-id"
}
```

#### Get Operation Details

```json
{
  "command": "get_operation",
  "operation_id": "your-operation-id"
}
```

**Response**:
```json
{
  "operation": {
    "operation_id": "...",
    "operation_type": "orchestration",
    "started_at": 1735734000.123,
    "status": "running",
    "progress": 0.75,
    "metadata": {},
    "event_count": 12
  }
}
```

#### Get Server Statistics

```json
{
  "command": "get_stats"
}
```

**Response**:
```json
{
  "stats": {
    "connected_clients": 5,
    "active_operations": 3,
    "total_operations": 47,
    "subscriptions": 8
  }
}
```

### Server API (Python)

#### Create Operation

```python
from mcp_hub.websocket_server import ws_manager

operation_id = await ws_manager.create_operation(
    "code_generation",
    metadata={"user_request": "Create a sorting function"}
)
```

#### Emit Progress

```python
await ws_manager.emit_progress(
    operation_id,
    progress=0.5,
    message="Halfway done",
    details={"step": "validation"}
)
```

#### Emit Status

```python
await ws_manager.emit_status(
    operation_id,
    status="processing",
    message="Processing data"
)
```

#### Emit Log

```python
await ws_manager.emit_log(
    operation_id,
    level="info",
    message="Operation started"
)
```

#### Emit Result

```python
await ws_manager.emit_result(
    operation_id,
    result={"output": "Success"}
)
```

#### Emit Error

```python
await ws_manager.emit_error(
    operation_id,
    error="Something went wrong",
    error_type="RuntimeError"
)
```

## Integration with Agents

### Using StreamingMixin

Add real-time streaming to custom agents:

```python
from mcp_hub.streaming_support import StreamingMixin

class MyCustomAgent(StreamingMixin):
    async def process_data(self, data):
        async with self.stream_operation('process_data', {'input': data}):
            # Step 1
            await self.emit_progress(0.2, "Loading data")
            loaded_data = await self.load(data)

            # Step 2
            await self.emit_progress(0.5, "Processing")
            processed = await self.process(loaded_data)

            # Step 3
            await self.emit_progress(0.8, "Finalizing")
            result = await self.finalize(processed)

            # Done
            await self.emit_result(result)
            return result
```

### Using ProgressTracker

For multi-step operations:

```python
from mcp_hub.streaming_support import ProgressTracker

async def multi_step_operation():
    from mcp_hub.websocket_server import ws_manager

    operation_id = await ws_manager.create_operation("multi_step")
    tracker = ProgressTracker(operation_id, total_steps=5)

    await tracker.step("Step 1: Initialize")
    # ... do work ...

    await tracker.step("Step 2: Process")
    # ... do work ...

    await tracker.step("Step 3: Validate")
    # ... do work ...

    await tracker.step("Step 4: Generate output")
    # ... do work ...

    await tracker.step("Step 5: Cleanup")
    result = await finalize()

    await tracker.complete(result)
    return result
```

### Direct Integration

For existing agents without mixins:

```python
from mcp_hub.websocket_server import ws_manager

async def my_operation():
    # Create operation
    operation_id = await ws_manager.create_operation("my_operation")

    try:
        # Progress updates
        await ws_manager.emit_progress(operation_id, 0.1, "Starting")
        # ... work ...

        await ws_manager.emit_progress(operation_id, 0.5, "Halfway")
        # ... more work ...

        # Success
        result = {"output": "done"}
        await ws_manager.emit_result(operation_id, result)
        return result

    except Exception as e:
        # Error
        await ws_manager.emit_error(operation_id, str(e), type(e).__name__)
        raise
```

## Best Practices

### 1. Use Meaningful Progress Values

Progress should represent actual completion:

```python
# Good: Based on actual steps
total_items = len(items)
for i, item in enumerate(items):
    progress = (i + 1) / total_items
    await ws_manager.emit_progress(op_id, progress, f"Processing item {i+1}/{total_items}")

# Bad: Arbitrary values
await ws_manager.emit_progress(op_id, 0.3, "Doing something")
await ws_manager.emit_progress(op_id, 0.6, "Doing something else")
```

### 2. Provide Descriptive Messages

```python
# Good: Specific and informative
await ws_manager.emit_progress(0.5, "Generated 150/300 lines of code")

# Bad: Vague
await ws_manager.emit_progress(0.5, "Working")
```

### 3. Include Relevant Details

```python
# Good: Helpful context
await ws_manager.emit_progress(
    0.75,
    "Code validation complete",
    details={
        "lines_validated": 250,
        "issues_found": 0,
        "time_taken": 2.3
    }
)
```

### 4. Use Appropriate Event Types

- **Progress**: For completion updates (0.0-1.0)
- **Status**: For state changes
- **Log**: For informational messages
- **Result**: For final output
- **Error**: For failures

### 5. Handle Connection Errors Gracefully

```python
try:
    await ws_manager.emit_progress(...)
except Exception as e:
    logger.warning(f"Failed to emit progress: {e}")
    # Continue operation anyway
```

### 6. Clean Up Operations

```python
# Always emit either result or error
try:
    result = await do_work()
    await ws_manager.emit_result(op_id, result)
except Exception as e:
    await ws_manager.emit_error(op_id, str(e))
```

## Troubleshooting

### Issue: WebSocket Server Not Starting

**Symptoms**: No "WebSocket server started" log message

**Solutions**:
1. Check if websockets is installed:
   ```bash
   pip install websockets>=12.0
   ```

2. Verify enabled in config:
   ```python
   import os
   print(os.environ.get("ENABLE_WEBSOCKET", "true"))
   ```

3. Check port availability:
   ```bash
   lsof -i :8765  # or netstat -an | grep 8765
   ```

### Issue: Cannot Connect from Client

**Symptoms**: Connection refused or timeout

**Solutions**:
1. Verify server is running:
   ```bash
   curl -X POST http://localhost:7860/api/get_websocket_status_service \
     -H "Content-Type: application/json" \
     -d '{"data": []}'
   ```

2. Check firewall settings:
   ```bash
   # Allow port 8765
   sudo ufw allow 8765
   ```

3. Use correct URL:
   ```
   ✓ ws://localhost:8765
   ✓ ws://127.0.0.1:8765
   ✗ http://localhost:8765  # Wrong protocol
   ```

### Issue: No Events Received

**Symptoms**: Connection successful but no events

**Solutions**:
1. Verify subscription:
   ```python
   ws.send(json.dumps({
       "command": "subscribe",
       "operation_id": "correct-id"
   }))
   ```

2. Check operation ID is correct:
   ```python
   # Get operation ID from create_operation
   op_id = await ws_manager.create_operation(...)
   ```

3. Verify operation is emitting events:
   ```python
   await ws_manager.emit_progress(op_id, 0.5, "Test")
   ```

### Issue: Connection Drops

**Symptoms**: Frequent disconnections

**Solutions**:
1. Implement reconnection logic:
   ```javascript
   function connectWithRetry() {
       const ws = new WebSocket('ws://localhost:8765');

       ws.onclose = () => {
           setTimeout(connectWithRetry, 5000);
       };

       return ws;
   }
   ```

2. Send periodic keep-alive:
   ```python
   while True:
       try:
           await websocket.ping()
           await asyncio.sleep(30)
       except:
           break
   ```

### Issue: High Memory Usage

**Symptoms**: WebSocket server using too much memory

**Solutions**:
1. Limit stored operations (default: 1000)
2. Clear old operations periodically
3. Unsubscribe from completed operations

## Integration with Other Features

### With Prometheus Metrics

Track WebSocket metrics:

```python
from mcp_hub.prometheus_metrics import track_request

async def operation_with_metrics():
    with track_request('websocket', 'stream_operation'):
        operation_id = await ws_manager.create_operation(...)
        # ... operation logic ...
```

### With Advanced Monitoring

Combine with request tracing:

```python
from mcp_hub.advanced_monitoring import advanced_monitor

async def traced_websocket_operation():
    with advanced_monitor.trace_request('websocket_op'):
        operation_id = await ws_manager.create_operation(...)
        # Both systems track this
```

### With Caching

Cache operation results:

```python
from mcp_hub.cache_utils import cache_manager

async def cached_streaming_operation(input_data):
    # Check cache first
    cached = cache_manager.get(f"op:{input_data}")
    if cached:
        return cached

    # Stream operation
    operation_id = await ws_manager.create_operation(...)
    result = await perform_operation(operation_id, input_data)

    # Cache result
    cache_manager.set(f"op:{input_data}", result)
    return result
```

## Further Reading

- [Advanced Monitoring Guide](ADVANCED_MONITORING.md)
- [Prometheus Metrics](PROMETHEUS.md)
- [Redis Caching](REDIS_CACHE.md)
- [API Documentation](API_DOCUMENTATION.md)

## Support

For issues or questions:
- Check [Troubleshooting](#troubleshooting) section
- Review example clients in `examples/`
- Open an issue on GitHub
