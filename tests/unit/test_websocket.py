"""Unit tests for WebSocket functionality."""

import pytest
import json
from unittest.mock import AsyncMock
from mcp_hub.websocket_server import (
    WebSocketEvent,
    Operation,
    WebSocketManager,
    EventType,
    ws_manager
)
from mcp_hub.streaming_support import (
    StreamingMixin,
    ProgressTracker,
    create_progress_callback,
    create_log_callback
)


class TestWebSocketEvent:
    """Tests for WebSocketEvent class."""

    def test_event_creation(self):
        """Test creating a WebSocket event."""
        event = WebSocketEvent(
            event_type="progress",
            operation_id="test-123",
            data={"progress": 0.5, "message": "Halfway"}
        )

        assert event.event_type == "progress"
        assert event.operation_id == "test-123"
        assert event.data["progress"] == 0.5

    def test_event_to_json(self):
        """Test converting event to JSON."""
        event = WebSocketEvent(
            event_type="status",
            operation_id="test-456",
            data={"status": "running"}
        )

        json_str = event.to_json()
        data = json.loads(json_str)

        assert data["event_type"] == "status"
        assert data["operation_id"] == "test-456"
        assert data["data"]["status"] == "running"
        assert "timestamp" in data


class TestOperation:
    """Tests for Operation class."""

    def test_operation_creation(self):
        """Test creating an operation."""
        operation = Operation(
            operation_id="op-123",
            operation_type="code_generation",
            started_at=123.456,
            metadata={"user": "test"}
        )

        assert operation.operation_id == "op-123"
        assert operation.operation_type == "code_generation"
        assert operation.status == "running"
        assert operation.progress == 0.0

    def test_operation_to_dict(self):
        """Test converting operation to dictionary."""
        operation = Operation(
            operation_id="op-456",
            operation_type="orchestration",
            started_at=123.456
        )

        data = operation.to_dict()

        assert data["operation_id"] == "op-456"
        assert data["operation_type"] == "orchestration"
        assert data["status"] == "running"
        assert data["progress"] == 0.0


class TestWebSocketManager:
    """Tests for WebSocketManager class."""

    @pytest.mark.asyncio
    async def test_manager_initialization(self):
        """Test WebSocket manager initialization."""
        manager = WebSocketManager()

        assert len(manager.connections) == 0
        assert len(manager.operations) == 0
        assert len(manager.operation_subscribers) == 0

    @pytest.mark.asyncio
    async def test_register_connection(self):
        """Test registering a WebSocket connection."""
        manager = WebSocketManager()
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()

        await manager.register(mock_ws)

        assert len(manager.connections) == 1
        assert mock_ws in manager.connections
        mock_ws.send.assert_called_once()  # Welcome message sent

    @pytest.mark.asyncio
    async def test_unregister_connection(self):
        """Test unregistering a WebSocket connection."""
        manager = WebSocketManager()
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()

        await manager.register(mock_ws)
        await manager.unregister(mock_ws)

        assert len(manager.connections) == 0

    @pytest.mark.asyncio
    async def test_create_operation(self):
        """Test creating an operation."""
        manager = WebSocketManager()

        operation_id = await manager.create_operation(
            "test_operation",
            {"key": "value"}
        )

        assert operation_id in manager.operations
        operation = manager.operations[operation_id]
        assert operation.operation_type == "test_operation"
        assert operation.metadata["key"] == "value"

    @pytest.mark.asyncio
    async def test_emit_progress(self):
        """Test emitting progress updates."""
        manager = WebSocketManager()
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()

        await manager.register(mock_ws)
        operation_id = await manager.create_operation("test")

        await manager.subscribe_to_operation(mock_ws, operation_id)
        await manager.emit_progress(operation_id, 0.5, "Halfway done")

        # Check that operation progress was updated
        operation = manager.operations[operation_id]
        assert operation.progress == 0.5

        # Check that message was sent
        assert mock_ws.send.call_count >= 2  # Welcome + operation created + progress

    @pytest.mark.asyncio
    async def test_emit_status(self):
        """Test emitting status updates."""
        manager = WebSocketManager()
        operation_id = await manager.create_operation("test")

        await manager.emit_status(operation_id, "processing", "Processing data")

        operation = manager.operations[operation_id]
        assert operation.status == "processing"

    @pytest.mark.asyncio
    async def test_emit_result(self):
        """Test emitting results."""
        manager = WebSocketManager()
        operation_id = await manager.create_operation("test")

        result = {"output": "test result"}
        await manager.emit_result(operation_id, result)

        operation = manager.operations[operation_id]
        assert operation.status == "completed"
        assert operation.progress == 1.0

    @pytest.mark.asyncio
    async def test_emit_error(self):
        """Test emitting errors."""
        manager = WebSocketManager()
        operation_id = await manager.create_operation("test")

        await manager.emit_error(operation_id, "Test error", "ValueError")

        operation = manager.operations[operation_id]
        assert operation.status == "error"

    @pytest.mark.asyncio
    async def test_subscribe_to_operation(self):
        """Test subscribing to an operation."""
        manager = WebSocketManager()
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()

        await manager.register(mock_ws)
        operation_id = await manager.create_operation("test")

        await manager.subscribe_to_operation(mock_ws, operation_id)

        assert operation_id in manager.operation_subscribers
        assert mock_ws in manager.operation_subscribers[operation_id]

    @pytest.mark.asyncio
    async def test_get_operation(self):
        """Test getting operation details."""
        manager = WebSocketManager()
        operation_id = await manager.create_operation("test", {"meta": "data"})

        operation_dict = manager.get_operation(operation_id)

        assert operation_dict is not None
        assert operation_dict["operation_id"] == operation_id
        assert operation_dict["operation_type"] == "test"

    @pytest.mark.asyncio
    async def test_get_active_operations(self):
        """Test getting active operations."""
        manager = WebSocketManager()

        # Create some operations
        op1 = await manager.create_operation("test1")
        op2 = await manager.create_operation("test2")

        # Complete one
        await manager.emit_result(op1, {"done": True})

        active = manager.get_active_operations()

        # Only op2 should be active
        assert len(active) == 1
        assert active[0]["operation_id"] == op2

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting server statistics."""
        manager = WebSocketManager()
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()

        await manager.register(mock_ws)
        operation_id = await manager.create_operation("test")
        await manager.subscribe_to_operation(mock_ws, operation_id)

        stats = manager.get_stats()

        assert stats["connected_clients"] == 1
        assert stats["total_operations"] == 1
        assert stats["subscriptions"] == 1


class TestStreamingMixin:
    """Tests for StreamingMixin class."""

    class TestAgent(StreamingMixin):
        """Test agent with streaming support."""
        pass

    @pytest.mark.asyncio
    async def test_stream_operation_context(self):
        """Test streaming operation context manager."""
        agent = self.TestAgent()

        async with agent.stream_operation("test_op", {"key": "value"}):
            # operation_id should be set
            assert agent._current_operation_id is not None

        # Should be cleared after context
        assert agent._current_operation_id is None

    @pytest.mark.asyncio
    async def test_emit_progress(self):
        """Test emitting progress from agent."""
        agent = self.TestAgent()

        async with agent.stream_operation("test_op"):
            # Should not raise even if WebSocket not available
            await agent.emit_progress(0.5, "Halfway")

    @pytest.mark.asyncio
    async def test_emit_status(self):
        """Test emitting status from agent."""
        agent = self.TestAgent()

        async with agent.stream_operation("test_op"):
            await agent.emit_status("processing", "Processing data")

    @pytest.mark.asyncio
    async def test_emit_log(self):
        """Test emitting log messages from agent."""
        agent = self.TestAgent()

        async with agent.stream_operation("test_op"):
            await agent.emit_log("info", "Processing started")

    @pytest.mark.asyncio
    async def test_emit_result(self):
        """Test emitting results from agent."""
        agent = self.TestAgent()

        async with agent.stream_operation("test_op"):
            await agent.emit_result({"output": "test"})


class TestProgressTracker:
    """Tests for ProgressTracker class."""

    @pytest.mark.asyncio
    async def test_progress_tracker_creation(self):
        """Test creating a progress tracker."""
        tracker = ProgressTracker("op-123", total_steps=5)

        assert tracker.operation_id == "op-123"
        assert tracker.total_steps == 5
        assert tracker.current_step == 0

    @pytest.mark.asyncio
    async def test_progress_tracker_step(self):
        """Test advancing progress tracker steps."""
        tracker = ProgressTracker("op-123", total_steps=4)

        await tracker.step("Step 1")
        assert tracker.current_step == 1

        await tracker.step("Step 2")
        assert tracker.current_step == 2

    @pytest.mark.asyncio
    async def test_progress_tracker_complete(self):
        """Test completing progress tracker."""
        tracker = ProgressTracker("op-123", total_steps=3)

        await tracker.step("Step 1")
        await tracker.step("Step 2")
        await tracker.step("Step 3")
        await tracker.complete({"result": "done"})

        assert tracker.current_step == 3


class TestCallbackCreators:
    """Tests for callback creator functions."""

    @pytest.mark.asyncio
    async def test_create_progress_callback(self):
        """Test creating progress callback."""
        callback = create_progress_callback("op-123")

        # Should not raise
        await callback(0.5, "Progress message")

    @pytest.mark.asyncio
    async def test_create_log_callback(self):
        """Test creating log callback."""
        callback = create_log_callback("op-123")

        # Should not raise
        await callback("info", "Log message")


class TestGlobalManager:
    """Tests for global WebSocket manager."""

    def test_global_manager_exists(self):
        """Test that global manager is initialized."""
        assert ws_manager is not None
        assert isinstance(ws_manager, WebSocketManager)


class TestEventType:
    """Tests for EventType enum."""

    def test_event_types(self):
        """Test EventType enum values."""
        assert EventType.CONNECTED.value == "connected"
        assert EventType.PROGRESS.value == "progress"
        assert EventType.STATUS.value == "status"
        assert EventType.RESULT.value == "result"
        assert EventType.ERROR.value == "error"
        assert EventType.COMPLETED.value == "completed"
        assert EventType.LOG.value == "log"
