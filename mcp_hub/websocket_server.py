"""WebSocket server for real-time streaming of long operations.

This module provides WebSocket support for streaming progress updates during
long-running operations like code generation, web search, and orchestration.
Clients can connect to receive real-time updates instead of polling.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, Set, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from .logging_config import logger

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logger.warning("websockets not available. Install with: pip install websockets>=12.0")


class EventType(Enum):
    """Types of WebSocket events."""
    CONNECTED = "connected"
    PROGRESS = "progress"
    STATUS = "status"
    RESULT = "result"
    ERROR = "error"
    COMPLETED = "completed"
    LOG = "log"


@dataclass
class WebSocketEvent:
    """A WebSocket event to be sent to clients."""

    event_type: str
    operation_id: str
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps({
            "event_type": self.event_type,
            "operation_id": self.operation_id,
            "timestamp": self.timestamp,
            "data": self.data
        })


@dataclass
class Operation:
    """A long-running operation being tracked."""

    operation_id: str
    operation_type: str
    started_at: float
    status: str = "running"
    progress: float = 0.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    events: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "started_at": self.started_at,
            "status": self.status,
            "progress": self.progress,
            "metadata": self.metadata,
            "event_count": len(self.events)
        }


class WebSocketManager:
    """Manages WebSocket connections and message broadcasting."""

    def __init__(self):
        """Initialize WebSocket manager."""
        self.connections: Set[WebSocketServerProtocol] = set()
        self.operations: Dict[str, Operation] = {}
        self.operation_subscribers: Dict[str, Set[WebSocketServerProtocol]] = {}
        self.lock = asyncio.Lock()

        logger.info("WebSocket manager initialized")

    async def register(self, websocket: WebSocketServerProtocol):
        """Register a new WebSocket connection.

        Args:
            websocket: WebSocket connection to register
        """
        async with self.lock:
            self.connections.add(websocket)
            logger.info(f"WebSocket client connected. Total: {len(self.connections)}")

        # Send welcome message
        welcome_event = WebSocketEvent(
            event_type=EventType.CONNECTED.value,
            operation_id="system",
            data={
                "message": "Connected to MCP Hub WebSocket server",
                "client_id": id(websocket),
                "timestamp": datetime.now().isoformat()
            }
        )
        await self._send_to_client(websocket, welcome_event)

    async def unregister(self, websocket: WebSocketServerProtocol):
        """Unregister a WebSocket connection.

        Args:
            websocket: WebSocket connection to unregister
        """
        async with self.lock:
            self.connections.discard(websocket)

            # Remove from operation subscribers
            for subscribers in self.operation_subscribers.values():
                subscribers.discard(websocket)

            logger.info(f"WebSocket client disconnected. Total: {len(self.connections)}")

    async def subscribe_to_operation(
        self,
        websocket: WebSocketServerProtocol,
        operation_id: str
    ):
        """Subscribe a client to updates for a specific operation.

        Args:
            websocket: WebSocket connection
            operation_id: Operation ID to subscribe to
        """
        async with self.lock:
            if operation_id not in self.operation_subscribers:
                self.operation_subscribers[operation_id] = set()

            self.operation_subscribers[operation_id].add(websocket)
            logger.debug(f"Client subscribed to operation: {operation_id}")

    async def create_operation(
        self,
        operation_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new operation and return its ID.

        Args:
            operation_type: Type of operation (e.g., 'orchestrate', 'code_generation')
            metadata: Optional metadata for the operation

        Returns:
            Operation ID
        """
        operation_id = str(uuid.uuid4())

        operation = Operation(
            operation_id=operation_id,
            operation_type=operation_type,
            started_at=time.time(),
            metadata=metadata or {}
        )

        async with self.lock:
            self.operations[operation_id] = operation

        # Broadcast operation created event
        event = WebSocketEvent(
            event_type=EventType.STATUS.value,
            operation_id=operation_id,
            data={
                "status": "started",
                "operation_type": operation_type,
                "metadata": metadata
            }
        )
        await self.broadcast_to_operation(operation_id, event)

        return operation_id

    async def emit_progress(
        self,
        operation_id: str,
        progress: float,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Emit a progress update for an operation.

        Args:
            operation_id: Operation ID
            progress: Progress value (0.0 to 1.0)
            message: Progress message
            details: Optional additional details
        """
        # Update operation
        async with self.lock:
            if operation_id in self.operations:
                self.operations[operation_id].progress = progress

        # Broadcast progress event
        event = WebSocketEvent(
            event_type=EventType.PROGRESS.value,
            operation_id=operation_id,
            data={
                "progress": progress,
                "message": message,
                "details": details or {}
            }
        )
        await self.broadcast_to_operation(operation_id, event)

    async def emit_status(
        self,
        operation_id: str,
        status: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Emit a status update for an operation.

        Args:
            operation_id: Operation ID
            status: Status value (e.g., 'running', 'completed', 'error')
            message: Status message
            details: Optional additional details
        """
        # Update operation
        async with self.lock:
            if operation_id in self.operations:
                self.operations[operation_id].status = status

        # Broadcast status event
        event = WebSocketEvent(
            event_type=EventType.STATUS.value,
            operation_id=operation_id,
            data={
                "status": status,
                "message": message,
                "details": details or {}
            }
        )
        await self.broadcast_to_operation(operation_id, event)

    async def emit_log(
        self,
        operation_id: str,
        level: str,
        message: str
    ):
        """Emit a log message for an operation.

        Args:
            operation_id: Operation ID
            level: Log level (info, warning, error)
            message: Log message
        """
        event = WebSocketEvent(
            event_type=EventType.LOG.value,
            operation_id=operation_id,
            data={
                "level": level,
                "message": message
            }
        )
        await self.broadcast_to_operation(operation_id, event)

    async def emit_result(
        self,
        operation_id: str,
        result: Any
    ):
        """Emit the final result of an operation.

        Args:
            operation_id: Operation ID
            result: Operation result
        """
        # Update operation
        async with self.lock:
            if operation_id in self.operations:
                self.operations[operation_id].status = "completed"
                self.operations[operation_id].progress = 1.0

        # Broadcast result event
        event = WebSocketEvent(
            event_type=EventType.RESULT.value,
            operation_id=operation_id,
            data={"result": result}
        )
        await self.broadcast_to_operation(operation_id, event)

        # Send completed event
        completed_event = WebSocketEvent(
            event_type=EventType.COMPLETED.value,
            operation_id=operation_id,
            data={
                "status": "completed",
                "duration": time.time() - self.operations[operation_id].started_at
            }
        )
        await self.broadcast_to_operation(operation_id, completed_event)

    async def emit_error(
        self,
        operation_id: str,
        error: str,
        error_type: Optional[str] = None
    ):
        """Emit an error for an operation.

        Args:
            operation_id: Operation ID
            error: Error message
            error_type: Optional error type
        """
        # Update operation
        async with self.lock:
            if operation_id in self.operations:
                self.operations[operation_id].status = "error"

        # Broadcast error event
        event = WebSocketEvent(
            event_type=EventType.ERROR.value,
            operation_id=operation_id,
            data={
                "error": error,
                "error_type": error_type or "UnknownError"
            }
        )
        await self.broadcast_to_operation(operation_id, event)

    async def broadcast_to_operation(
        self,
        operation_id: str,
        event: WebSocketEvent
    ):
        """Broadcast an event to all subscribers of an operation.

        Args:
            operation_id: Operation ID
            event: Event to broadcast
        """
        async with self.lock:
            subscribers = self.operation_subscribers.get(operation_id, set()).copy()

            # Store event in operation
            if operation_id in self.operations:
                self.operations[operation_id].events.append(event)

        # Send to all subscribers
        if subscribers:
            await asyncio.gather(
                *[self._send_to_client(ws, event) for ws in subscribers],
                return_exceptions=True
            )

    async def broadcast_to_all(self, event: WebSocketEvent):
        """Broadcast an event to all connected clients.

        Args:
            event: Event to broadcast
        """
        async with self.lock:
            connections = self.connections.copy()

        if connections:
            await asyncio.gather(
                *[self._send_to_client(ws, event) for ws in connections],
                return_exceptions=True
            )

    async def _send_to_client(
        self,
        websocket: WebSocketServerProtocol,
        event: WebSocketEvent
    ):
        """Send an event to a specific client.

        Args:
            websocket: WebSocket connection
            event: Event to send
        """
        try:
            await websocket.send(event.to_json())
        except Exception as e:
            logger.error(f"Failed to send to client: {e}")
            await self.unregister(websocket)

    def get_operation(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get operation details.

        Args:
            operation_id: Operation ID

        Returns:
            Operation details or None
        """
        operation = self.operations.get(operation_id)
        return operation.to_dict() if operation else None

    def get_active_operations(self) -> list:
        """Get all active operations.

        Returns:
            List of active operation details
        """
        return [
            op.to_dict() for op in self.operations.values()
            if op.status == "running"
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket server statistics.

        Returns:
            Dictionary with server stats
        """
        return {
            "connected_clients": len(self.connections),
            "active_operations": len([
                op for op in self.operations.values()
                if op.status == "running"
            ]),
            "total_operations": len(self.operations),
            "subscriptions": sum(len(subs) for subs in self.operation_subscribers.values())
        }


# Global WebSocket manager instance
ws_manager = WebSocketManager()


async def handle_websocket(websocket: WebSocketServerProtocol, path: str):
    """Handle WebSocket connections.

    Args:
        websocket: WebSocket connection
        path: Connection path
    """
    await ws_manager.register(websocket)

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                command = data.get("command")

                if command == "subscribe":
                    operation_id = data.get("operation_id")
                    if operation_id:
                        await ws_manager.subscribe_to_operation(websocket, operation_id)
                        response = {
                            "status": "subscribed",
                            "operation_id": operation_id
                        }
                        await websocket.send(json.dumps(response))

                elif command == "get_operation":
                    operation_id = data.get("operation_id")
                    operation = ws_manager.get_operation(operation_id)
                    await websocket.send(json.dumps({
                        "operation": operation
                    }))

                elif command == "get_stats":
                    stats = ws_manager.get_stats()
                    await websocket.send(json.dumps({
                        "stats": stats
                    }))

                else:
                    await websocket.send(json.dumps({
                        "error": f"Unknown command: {command}"
                    }))

            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "error": "Invalid JSON"
                }))
            except Exception as e:
                logger.error(f"Error handling message: {e}")
                await websocket.send(json.dumps({
                    "error": str(e)
                }))

    except websockets.exceptions.ConnectionClosed:
        logger.debug("WebSocket connection closed")

    finally:
        await ws_manager.unregister(websocket)


async def start_websocket_server(host: str = "0.0.0.0", port: int = 8765):
    """Start the WebSocket server.

    Args:
        host: Host to bind to
        port: Port to bind to
    """
    if not WEBSOCKETS_AVAILABLE:
        logger.error("WebSocket server cannot start: websockets library not available")
        return

    async with websockets.serve(handle_websocket, host, port):
        logger.info(f"WebSocket server started on ws://{host}:{port}")
        await asyncio.Future()  # Run forever
