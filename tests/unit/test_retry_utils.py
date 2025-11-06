"""Unit tests for retry utilities."""

import pytest
import time
import asyncio
from unittest.mock import Mock
from mcp_hub.retry_utils import (
    exponential_backoff,
    retry_sync,
    retry_async,
    should_retry_exception,
    retry_with_filter,
    RetryConfig
)


class TestExponentialBackoff:
    """Tests for exponential backoff calculation."""

    def test_exponential_backoff_increases(self):
        """Test that backoff increases exponentially."""
        delay0 = exponential_backoff(0, base_delay=1.0, jitter=False)
        delay1 = exponential_backoff(1, base_delay=1.0, jitter=False)
        delay2 = exponential_backoff(2, base_delay=1.0, jitter=False)

        assert delay0 == 1.0
        assert delay1 == 2.0
        assert delay2 == 4.0

    def test_exponential_backoff_max_delay(self):
        """Test that backoff respects max_delay."""
        delay = exponential_backoff(10, base_delay=1.0, max_delay=30.0, jitter=False)
        assert delay == 30.0

    def test_exponential_backoff_with_jitter(self):
        """Test that jitter adds randomness."""
        delays = [exponential_backoff(1, base_delay=1.0, jitter=True) for _ in range(10)]
        # All delays should be different with jitter
        assert len(set(delays)) > 1
        # All should be between 1.0 and 2.0
        assert all(1.0 <= d <= 2.0 for d in delays)


class TestRetrySyncDecorator:
    """Tests for synchronous retry decorator."""

    def test_retry_sync_success_first_attempt(self):
        """Test that successful function doesn't retry."""
        call_count = [0]

        @retry_sync(max_attempts=3, base_delay=0.01)
        def succeeds():
            call_count[0] += 1
            return "success"

        result = succeeds()
        assert result == "success"
        assert call_count[0] == 1

    def test_retry_sync_fails_then_succeeds(self):
        """Test that function retries on failure and succeeds."""
        call_count = [0]

        @retry_sync(max_attempts=3, base_delay=0.01)
        def fails_twice():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = fails_twice()
        assert result == "success"
        assert call_count[0] == 3

    def test_retry_sync_max_attempts_exceeded(self):
        """Test that function raises after max attempts."""
        call_count = [0]

        @retry_sync(max_attempts=3, base_delay=0.01)
        def always_fails():
            call_count[0] += 1
            raise ConnectionError("Permanent failure")

        with pytest.raises(ConnectionError):
            always_fails()

        assert call_count[0] == 3

    def test_retry_sync_specific_exception(self):
        """Test retry only on specific exceptions."""
        call_count = [0]

        @retry_sync(max_attempts=3, base_delay=0.01, exceptions=(ConnectionError,))
        def raises_value_error():
            call_count[0] += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            raises_value_error()

        # Should not retry ValueError
        assert call_count[0] == 1

    def test_retry_sync_on_retry_callback(self):
        """Test that on_retry callback is called."""
        callback_calls = []

        def on_retry(attempt, error, delay):
            callback_calls.append((attempt, str(error), delay))

        call_count = [0]

        @retry_sync(max_attempts=3, base_delay=0.01, on_retry=on_retry)
        def fails_twice():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError(f"Attempt {call_count[0]}")
            return "success"

        result = fails_twice()
        assert result == "success"
        assert len(callback_calls) == 2  # Called on first two failures


class TestRetryAsyncDecorator:
    """Tests for asynchronous retry decorator."""

    @pytest.mark.asyncio
    async def test_retry_async_success_first_attempt(self):
        """Test that successful async function doesn't retry."""
        call_count = [0]

        @retry_async(max_attempts=3, base_delay=0.01)
        async def succeeds():
            call_count[0] += 1
            return "success"

        result = await succeeds()
        assert result == "success"
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_retry_async_fails_then_succeeds(self):
        """Test that async function retries on failure and succeeds."""
        call_count = [0]

        @retry_async(max_attempts=3, base_delay=0.01)
        async def fails_twice():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = await fails_twice()
        assert result == "success"
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_retry_async_max_attempts_exceeded(self):
        """Test that async function raises after max attempts."""
        call_count = [0]

        @retry_async(max_attempts=3, base_delay=0.01)
        async def always_fails():
            call_count[0] += 1
            raise ConnectionError("Permanent failure")

        with pytest.raises(ConnectionError):
            await always_fails()

        assert call_count[0] == 3


class TestShouldRetryException:
    """Tests for exception retry filter."""

    def test_should_retry_connection_error(self):
        """Test that connection errors are retryable."""
        assert should_retry_exception(ConnectionError("test"))

    def test_should_retry_timeout_error(self):
        """Test that timeout errors are retryable."""
        assert should_retry_exception(TimeoutError("test"))

    def test_should_retry_rate_limit(self):
        """Test that rate limit errors are retryable."""
        assert should_retry_exception(Exception("429 Too Many Requests"))
        assert should_retry_exception(Exception("Rate limit exceeded"))

    def test_should_retry_service_unavailable(self):
        """Test that 503 errors are retryable."""
        assert should_retry_exception(Exception("503 Service Unavailable"))

    def test_should_not_retry_value_error(self):
        """Test that value errors are not retryable."""
        assert not should_retry_exception(ValueError("Invalid input"))

    def test_should_not_retry_type_error(self):
        """Test that type errors are not retryable."""
        assert not should_retry_exception(TypeError("Wrong type"))


class TestRetryWithFilter:
    """Tests for filtered retry decorator."""

    def test_retry_with_filter_retries_transient(self):
        """Test that filtered retry retries transient errors."""
        call_count = [0]

        @retry_with_filter(max_attempts=3, base_delay=0.01)
        def transient_failure():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Timeout")
            return "success"

        result = transient_failure()
        assert result == "success"
        assert call_count[0] == 3

    def test_retry_with_filter_does_not_retry_permanent(self):
        """Test that filtered retry doesn't retry permanent errors."""
        call_count = [0]

        @retry_with_filter(max_attempts=3, base_delay=0.01)
        def permanent_failure():
            call_count[0] += 1
            raise ValueError("Invalid input")

        with pytest.raises(ValueError):
            permanent_failure()

        # Should not retry ValueError
        assert call_count[0] == 1


class TestRetryConfig:
    """Tests for retry configuration."""

    def test_retry_config_has_all_configs(self):
        """Test that RetryConfig has all expected configurations."""
        assert hasattr(RetryConfig, 'LLM_API')
        assert hasattr(RetryConfig, 'SEARCH_API')
        assert hasattr(RetryConfig, 'CODE_EXECUTION')
        assert hasattr(RetryConfig, 'NETWORK')

    def test_retry_config_llm_api(self):
        """Test LLM API retry configuration."""
        config = RetryConfig.LLM_API
        assert 'max_attempts' in config
        assert 'base_delay' in config
        assert 'max_delay' in config
        assert config['max_attempts'] >= 2

    def test_retry_config_search_api(self):
        """Test Search API retry configuration."""
        config = RetryConfig.SEARCH_API
        assert 'max_attempts' in config
        assert 'base_delay' in config
        assert config['max_attempts'] >= 2
