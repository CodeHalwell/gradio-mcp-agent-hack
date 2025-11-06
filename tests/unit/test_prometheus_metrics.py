"""Unit tests for Prometheus metrics module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from mcp_hub.prometheus_metrics import (
    update_system_metrics,
    get_prometheus_metrics,
    get_prometheus_content_type,
    track_request,
    track_llm_call,
    track_search_call,
    track_code_execution,
    track_cache_access,
    track_retry,
    get_metrics_summary,
)


class TestSystemMetrics:
    """Tests for system metrics collection."""

    @patch('mcp_hub.prometheus_metrics.psutil')
    def test_update_system_metrics(self, mock_psutil):
        """Test that system metrics are updated correctly."""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 45.5
        mock_memory = Mock()
        mock_memory.percent = 60.0
        mock_memory.available = 4 * 1024 * 1024 * 1024  # 4GB
        mock_psutil.virtual_memory.return_value = mock_memory

        mock_disk = Mock()
        mock_disk.percent = 70.0
        mock_disk.free = 100 * 1024 * 1024 * 1024  # 100GB
        mock_psutil.disk_usage.return_value = mock_disk

        # Update metrics
        update_system_metrics()

        # Verify psutil was called
        mock_psutil.cpu_percent.assert_called_once()
        mock_psutil.virtual_memory.assert_called_once()
        mock_psutil.disk_usage.assert_called_once()

    @patch('mcp_hub.prometheus_metrics.psutil')
    def test_update_system_metrics_handles_errors(self, mock_psutil):
        """Test that errors in system metrics collection are handled gracefully."""
        mock_psutil.cpu_percent.side_effect = Exception("Test error")

        # Should not raise
        update_system_metrics()


class TestPrometheusExport:
    """Tests for Prometheus format export."""

    def test_get_prometheus_metrics_returns_bytes(self):
        """Test that Prometheus metrics are returned as bytes."""
        result = get_prometheus_metrics()
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_get_prometheus_metrics_contains_app_info(self):
        """Test that Prometheus metrics contain application info."""
        result = get_prometheus_metrics().decode('utf-8')
        assert 'mcp_hub_app' in result

    def test_get_prometheus_content_type(self):
        """Test that content type is correct."""
        content_type = get_prometheus_content_type()
        assert 'text/plain' in content_type

    @patch('mcp_hub.prometheus_metrics.psutil')
    def test_prometheus_metrics_include_system_metrics(self, mock_psutil):
        """Test that Prometheus export includes system metrics."""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 45.5
        mock_memory = Mock()
        mock_memory.percent = 60.0
        mock_memory.available = 4 * 1024 * 1024 * 1024
        mock_psutil.virtual_memory.return_value = mock_memory

        mock_disk = Mock()
        mock_disk.percent = 70.0
        mock_disk.free = 100 * 1024 * 1024 * 1024
        mock_psutil.disk_usage.return_value = mock_disk

        result = get_prometheus_metrics().decode('utf-8')

        # Check that system metrics are present
        assert 'mcp_hub_cpu_usage_percent' in result
        assert 'mcp_hub_memory_usage_percent' in result
        assert 'mcp_hub_disk_usage_percent' in result


class TestRequestTracking:
    """Tests for request tracking."""

    def test_track_request_success(self):
        """Test tracking a successful request."""
        with track_request('test_agent', 'test_operation'):
            pass  # Simulate successful operation

        # Verify metrics were recorded
        result = get_prometheus_metrics().decode('utf-8')
        assert 'mcp_hub_requests_total' in result

    def test_track_request_failure(self):
        """Test tracking a failed request."""
        try:
            with track_request('test_agent', 'test_operation'):
                raise ValueError("Test error")
        except ValueError:
            pass

        # Verify failure was recorded
        result = get_prometheus_metrics().decode('utf-8')
        assert 'mcp_hub_requests_total' in result
        assert 'mcp_hub_requests_failed_total' in result

    def test_track_request_duration(self):
        """Test that request duration is tracked."""
        import time

        with track_request('test_agent', 'test_operation'):
            time.sleep(0.01)  # Small delay

        result = get_prometheus_metrics().decode('utf-8')
        assert 'mcp_hub_request_duration_seconds' in result


class TestAPIMetrics:
    """Tests for API-specific metrics."""

    def test_track_llm_call(self):
        """Test tracking LLM API calls."""
        track_llm_call('openai', 'gpt-4')
        track_llm_call('anthropic', 'claude-3')

        result = get_prometheus_metrics().decode('utf-8')
        assert 'mcp_hub_llm_api_calls_total' in result

    def test_track_search_call(self):
        """Test tracking search API calls."""
        track_search_call()
        track_search_call()

        result = get_prometheus_metrics().decode('utf-8')
        assert 'mcp_hub_search_api_calls_total' in result

    def test_track_code_execution(self):
        """Test tracking code execution."""
        track_code_execution('success')
        track_code_execution('failure')

        result = get_prometheus_metrics().decode('utf-8')
        assert 'mcp_hub_code_executions_total' in result


class TestCacheMetrics:
    """Tests for cache metrics."""

    def test_track_cache_hit(self):
        """Test tracking cache hits."""
        track_cache_access('llm', hit=True)
        track_cache_access('llm', hit=True)

        result = get_prometheus_metrics().decode('utf-8')
        assert 'mcp_hub_cache_hits_total' in result

    def test_track_cache_miss(self):
        """Test tracking cache misses."""
        track_cache_access('search', hit=False)

        result = get_prometheus_metrics().decode('utf-8')
        assert 'mcp_hub_cache_misses_total' in result


class TestRetryMetrics:
    """Tests for retry metrics."""

    def test_track_retry(self):
        """Test tracking retry attempts."""
        track_retry('llm_api', attempt=1)
        track_retry('llm_api', attempt=2)

        result = get_prometheus_metrics().decode('utf-8')
        assert 'mcp_hub_retries_total' in result

    def test_track_retry_exhausted(self):
        """Test tracking exhausted retries."""
        track_retry('search_api', attempt=3, exhausted=True)

        result = get_prometheus_metrics().decode('utf-8')
        assert 'mcp_hub_retries_exhausted_total' in result


class TestMetricsSummary:
    """Tests for metrics summary."""

    @patch('mcp_hub.prometheus_metrics.psutil')
    def test_get_metrics_summary(self, mock_psutil):
        """Test getting metrics summary."""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 45.5
        mock_memory = Mock()
        mock_memory.percent = 60.0
        mock_memory.available = 4 * 1024 * 1024 * 1024
        mock_psutil.virtual_memory.return_value = mock_memory

        mock_disk = Mock()
        mock_disk.percent = 70.0
        mock_disk.free = 100 * 1024 * 1024 * 1024
        mock_psutil.disk_usage.return_value = mock_disk

        summary = get_metrics_summary()

        assert 'timestamp' in summary
        assert 'uptime_seconds' in summary
        assert 'system' in summary
        assert 'info' in summary
        assert summary['system']['cpu_percent'] == 45.5
        assert summary['system']['memory_percent'] == 60.0


class TestMetricsCollectorIntegration:
    """Tests for integration with existing metrics_collector."""

    @patch('mcp_hub.prometheus_metrics.METRICS_COLLECTOR_AVAILABLE', True)
    @patch('mcp_hub.prometheus_metrics.metrics_collector')
    def test_sync_from_metrics_collector(self, mock_collector):
        """Test syncing from existing metrics collector."""
        # Mock metrics collector
        mock_collector.get_metrics_summary.return_value = {
            'web_search_duration_seconds': {
                'count': 10,
                'average': 1.5,
                'min': 0.5,
                'max': 3.0,
                'latest': 1.2
            }
        }

        # This should not raise an error
        result = get_prometheus_metrics()
        assert isinstance(result, bytes)

    @patch('mcp_hub.prometheus_metrics.METRICS_COLLECTOR_AVAILABLE', False)
    def test_sync_without_metrics_collector(self):
        """Test that sync works when metrics collector is not available."""
        # Should not raise an error
        result = get_prometheus_metrics()
        assert isinstance(result, bytes)
