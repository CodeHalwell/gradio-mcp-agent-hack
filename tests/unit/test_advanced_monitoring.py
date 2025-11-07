"""Unit tests for advanced monitoring module."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from mcp_hub.advanced_monitoring import (
    AdvancedMonitor,
    RequestTrace,
    Span,
    SlowQuery,
    MemorySnapshot,
    advanced_monitor,
    trace_operation
)
from datetime import datetime


class TestRequestTrace:
    """Tests for RequestTrace class."""

    def test_request_trace_creation(self):
        """Test creating a request trace."""
        trace = RequestTrace(
            trace_id="test-123",
            operation="web_search",
            start_time=time.time(),
            metadata={"query": "python"}
        )

        assert trace.trace_id == "test-123"
        assert trace.operation == "web_search"
        assert trace.status == "running"
        assert trace.metadata["query"] == "python"

    def test_request_trace_complete(self):
        """Test completing a trace."""
        trace = RequestTrace(
            trace_id="test-123",
            operation="test",
            start_time=time.time()
        )

        time.sleep(0.1)
        trace.complete(status="success")

        assert trace.status == "success"
        assert trace.end_time is not None
        assert trace.duration > 0
        assert trace.duration >= 0.1

    def test_request_trace_complete_with_error(self):
        """Test completing a trace with error."""
        trace = RequestTrace(
            trace_id="test-123",
            operation="test",
            start_time=time.time()
        )

        trace.complete(status="failed", error="Test error")

        assert trace.status == "failed"
        assert trace.error == "Test error"

    def test_add_span(self):
        """Test adding spans to a trace."""
        trace = RequestTrace(
            trace_id="test-123",
            operation="test",
            start_time=time.time()
        )

        trace.add_span("search", 1.5, {"results": 10})
        trace.add_span("process", 0.5)

        assert len(trace.spans) == 2
        assert trace.spans[0].name == "search"
        assert trace.spans[0].duration == 1.5
        assert trace.spans[1].name == "process"

    def test_trace_to_dict(self):
        """Test converting trace to dictionary."""
        trace = RequestTrace(
            trace_id="test-123",
            operation="test",
            start_time=time.time(),
            metadata={"key": "value"}
        )
        trace.add_span("span1", 1.0)
        trace.complete()

        data = trace.to_dict()

        assert data["trace_id"] == "test-123"
        assert data["operation"] == "test"
        assert data["status"] == "success"
        assert data["metadata"]["key"] == "value"
        assert len(data["spans"]) == 1


class TestSpan:
    """Tests for Span class."""

    def test_span_creation(self):
        """Test creating a span."""
        span = Span(
            name="database_query",
            start_time=time.time(),
            duration=0.5,
            metadata={"table": "users"}
        )

        assert span.name == "database_query"
        assert span.duration == 0.5
        assert span.metadata["table"] == "users"

    def test_span_to_dict(self):
        """Test converting span to dictionary."""
        span = Span(
            name="test_span",
            start_time=time.time(),
            duration=1.0,
            metadata={"key": "value"}
        )

        data = span.to_dict()

        assert data["name"] == "test_span"
        assert data["duration"] == 1.0
        assert data["metadata"]["key"] == "value"


class TestSlowQuery:
    """Tests for SlowQuery class."""

    def test_slow_query_creation(self):
        """Test creating a slow query record."""
        slow_query = SlowQuery(
            operation="web_search",
            duration=10.5,
            timestamp=datetime.now(),
            trace_id="test-123",
            metadata={"query": "complex search"}
        )

        assert slow_query.operation == "web_search"
        assert slow_query.duration == 10.5
        assert slow_query.trace_id == "test-123"

    def test_slow_query_to_dict(self):
        """Test converting slow query to dictionary."""
        now = datetime.now()
        slow_query = SlowQuery(
            operation="test_op",
            duration=8.0,
            timestamp=now,
            trace_id="test-456"
        )

        data = slow_query.to_dict()

        assert data["operation"] == "test_op"
        assert data["duration"] == 8.0
        assert data["trace_id"] == "test-456"
        assert "timestamp" in data


class TestMemorySnapshot:
    """Tests for MemorySnapshot class."""

    def test_memory_snapshot_creation(self):
        """Test creating a memory snapshot."""
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            rss_mb=100.5,
            vms_mb=200.3,
            percent=45.2,
            available_mb=2048.0
        )

        assert snapshot.rss_mb == 100.5
        assert snapshot.percent == 45.2

    def test_memory_snapshot_to_dict(self):
        """Test converting memory snapshot to dictionary."""
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            rss_mb=100.0,
            vms_mb=200.0,
            percent=50.0,
            available_mb=2000.0
        )

        data = snapshot.to_dict()

        assert data["rss_mb"] == 100.0
        assert data["percent"] == 50.0
        assert "timestamp" in data


class TestAdvancedMonitor:
    """Tests for AdvancedMonitor class."""

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = AdvancedMonitor(
            slow_query_threshold=3.0,
            max_traces=500,
            memory_tracking_enabled=False  # Disable for testing
        )

        assert monitor.slow_query_threshold == 3.0
        assert monitor.max_traces == 500
        assert len(monitor.traces) == 0

    def test_trace_request_success(self):
        """Test tracing a successful request."""
        monitor = AdvancedMonitor(memory_tracking_enabled=False)

        with monitor.trace_request("test_operation", {"key": "value"}) as trace:
            time.sleep(0.1)
            # Operation succeeds

        assert len(monitor.traces) > 0
        completed_trace = monitor.traces[-1]
        assert completed_trace.status == "success"
        assert completed_trace.duration >= 0.1

    def test_trace_request_failure(self):
        """Test tracing a failed request."""
        monitor = AdvancedMonitor(memory_tracking_enabled=False)

        with pytest.raises(ValueError):
            with monitor.trace_request("test_operation"):
                raise ValueError("Test error")

        completed_trace = monitor.traces[-1]
        assert completed_trace.status == "failed"
        assert "Test error" in completed_trace.error

    def test_slow_query_detection(self):
        """Test slow query detection."""
        monitor = AdvancedMonitor(
            slow_query_threshold=0.1,
            memory_tracking_enabled=False
        )

        with monitor.trace_request("slow_operation"):
            time.sleep(0.2)  # Exceeds threshold

        assert len(monitor.slow_queries) > 0
        slow_query = monitor.slow_queries[-1]
        assert slow_query.operation == "slow_operation"
        assert slow_query.duration >= 0.2

    def test_trace_span(self):
        """Test tracing spans within a request."""
        monitor = AdvancedMonitor(memory_tracking_enabled=False)

        with monitor.trace_request("parent_operation") as trace:
            with monitor.trace_span(trace, "child_span_1"):
                time.sleep(0.05)

            with monitor.trace_span(trace, "child_span_2", {"meta": "data"}):
                time.sleep(0.05)

        completed_trace = monitor.traces[-1]
        assert len(completed_trace.spans) == 2
        assert completed_trace.spans[0].name == "child_span_1"
        assert completed_trace.spans[1].name == "child_span_2"
        assert completed_trace.spans[1].metadata.get("meta") == "data"

    def test_get_recent_traces(self):
        """Test getting recent traces."""
        monitor = AdvancedMonitor(memory_tracking_enabled=False)

        # Create some traces
        for i in range(5):
            with monitor.trace_request(f"operation_{i}"):
                pass

        traces = monitor.get_recent_traces(limit=3)

        assert len(traces) == 3
        # Should be in reverse order (most recent first)
        assert traces[0]["operation"] == "operation_4"

    def test_get_active_traces(self):
        """Test getting active traces."""
        monitor = AdvancedMonitor(memory_tracking_enabled=False)

        # This test is tricky because traces complete immediately
        # In real usage, active_traces would be populated during long-running ops
        with monitor.trace_request("long_operation") as trace:
            active = monitor.get_active_traces()
            assert len(active) > 0

    def test_get_operation_stats(self):
        """Test getting operation statistics."""
        monitor = AdvancedMonitor(memory_tracking_enabled=False)

        # Create multiple traces for same operation
        for _ in range(3):
            with monitor.trace_request("test_op"):
                time.sleep(0.05)

        # Create one failed trace
        try:
            with monitor.trace_request("test_op"):
                raise Exception("Test error")
        except:
            pass

        stats = monitor.get_operation_stats()

        assert "test_op" in stats
        op_stats = stats["test_op"]
        assert op_stats["count"] == 4
        assert op_stats["errors"] == 1
        assert op_stats["error_rate"] == 0.25
        assert op_stats["average_duration"] > 0

    @patch('mcp_hub.advanced_monitoring.psutil')
    def test_get_memory_stats(self, mock_psutil):
        """Test getting memory statistics."""
        # Create monitor without background thread
        monitor = AdvancedMonitor(memory_tracking_enabled=False)

        # Manually add some memory snapshots
        for i in range(5):
            snapshot = MemorySnapshot(
                timestamp=datetime.now(),
                rss_mb=100.0 + i,
                vms_mb=200.0,
                percent=50.0 + i,
                available_mb=2000.0
            )
            monitor.memory_snapshots.append(snapshot)

        stats = monitor.get_memory_stats(minutes=5)

        assert "current_rss_mb" in stats
        assert "average_percent" in stats
        assert stats["snapshots_count"] == 5

    def test_detect_bottlenecks_slow_operation(self):
        """Test bottleneck detection for slow operations."""
        monitor = AdvancedMonitor(
            slow_query_threshold=0.1,
            memory_tracking_enabled=False
        )

        # Create slow operations
        for _ in range(3):
            with monitor.trace_request("slow_op"):
                time.sleep(0.15)

        bottlenecks = monitor.detect_bottlenecks()

        assert len(bottlenecks) > 0
        slow_bottleneck = next(
            (b for b in bottlenecks if b["type"] == "slow_operation"),
            None
        )
        assert slow_bottleneck is not None
        assert slow_bottleneck["operation"] == "slow_op"

    def test_detect_bottlenecks_high_error_rate(self):
        """Test bottleneck detection for high error rates."""
        monitor = AdvancedMonitor(memory_tracking_enabled=False)

        # Create operations with high error rate
        for i in range(10):
            try:
                with monitor.trace_request("error_prone_op"):
                    if i % 2 == 0:  # 50% error rate
                        raise Exception("Test error")
            except:
                pass

        bottlenecks = monitor.detect_bottlenecks()

        error_bottleneck = next(
            (b for b in bottlenecks if b["type"] == "high_error_rate"),
            None
        )
        assert error_bottleneck is not None

    def test_reset_stats(self):
        """Test resetting statistics."""
        monitor = AdvancedMonitor(memory_tracking_enabled=False)

        # Create some data
        with monitor.trace_request("test"):
            pass

        assert len(monitor.traces) > 0

        # Reset
        monitor.reset_stats()

        assert len(monitor.traces) == 0
        assert len(monitor.operation_stats) == 0


class TestTraceOperationDecorator:
    """Tests for trace_operation decorator."""

    def test_decorator_sync_function(self):
        """Test decorator on synchronous function."""
        monitor = AdvancedMonitor(memory_tracking_enabled=False)

        @trace_operation("test_sync")
        def sync_function():
            time.sleep(0.05)
            return "result"

        # Temporarily replace global monitor
        import mcp_hub.advanced_monitoring
        old_monitor = mcp_hub.advanced_monitoring.advanced_monitor
        mcp_hub.advanced_monitoring.advanced_monitor = monitor

        try:
            result = sync_function()

            assert result == "result"
            assert len(monitor.traces) > 0
            assert monitor.traces[-1].operation == "test_sync"

        finally:
            mcp_hub.advanced_monitoring.advanced_monitor = old_monitor

    @pytest.mark.asyncio
    async def test_decorator_async_function(self):
        """Test decorator on asynchronous function."""
        import asyncio
        monitor = AdvancedMonitor(memory_tracking_enabled=False)

        @trace_operation("test_async")
        async def async_function():
            await asyncio.sleep(0.05)
            return "async_result"

        # Temporarily replace global monitor
        import mcp_hub.advanced_monitoring
        old_monitor = mcp_hub.advanced_monitoring.advanced_monitor
        mcp_hub.advanced_monitoring.advanced_monitor = monitor

        try:
            result = await async_function()

            assert result == "async_result"
            assert len(monitor.traces) > 0
            assert monitor.traces[-1].operation == "test_async"

        finally:
            mcp_hub.advanced_monitoring.advanced_monitor = old_monitor


class TestGlobalMonitor:
    """Tests for global monitor instance."""

    def test_global_monitor_exists(self):
        """Test that global monitor is initialized."""
        assert advanced_monitor is not None
        assert isinstance(advanced_monitor, AdvancedMonitor)

    def test_global_monitor_configured(self):
        """Test global monitor has default configuration."""
        assert advanced_monitor.slow_query_threshold == 5.0
        assert advanced_monitor.max_traces == 1000
