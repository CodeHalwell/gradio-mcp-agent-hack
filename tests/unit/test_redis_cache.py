"""Unit tests for Redis cache backend."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta


# Mock redis module for tests
class MockRedis:
    """Mock Redis client for testing."""

    def __init__(self):
        self.data = {}
        self.expirations = {}

    def ping(self):
        """Mock ping."""
        return True

    def get(self, key):
        """Mock get."""
        if key in self.data and key in self.expirations:
            if datetime.now() > self.expirations[key]:
                del self.data[key]
                del self.expirations[key]
                return None
        return self.data.get(key)

    def setex(self, key, ttl, value):
        """Mock setex."""
        self.data[key] = value
        self.expirations[key] = datetime.now() + timedelta(seconds=ttl)
        return True

    def delete(self, *keys):
        """Mock delete."""
        count = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                if key in self.expirations:
                    del self.expirations[key]
                count += 1
        return count

    def scan_iter(self, match="*", count=100):
        """Mock scan_iter."""
        pattern = match.replace("*", "")
        for key in self.data.keys():
            if key.startswith(pattern):
                yield key

    def scan(self, cursor, match="*", count=100):
        """Mock scan."""
        pattern = match.replace("*", "")
        keys = [k for k in self.data.keys() if k.startswith(pattern)]
        return 0, keys

    def info(self):
        """Mock info."""
        return {
            "redis_version": "7.0.0",
            "connected_clients": 5,
            "used_memory": 1024 * 1024,
            "used_memory_human": "1M",
        }


class MockConnectionPool:
    """Mock connection pool."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @staticmethod
    def from_url(url, **kwargs):
        """Mock from_url."""
        return MockConnectionPool(url=url, **kwargs)

    def disconnect(self):
        """Mock disconnect."""
        pass


@pytest.fixture
def mock_redis_module():
    """Mock the redis module."""
    mock_redis = MagicMock()
    mock_redis.Redis.return_value = MockRedis()
    mock_redis.ConnectionPool = MockConnectionPool
    mock_redis.ConnectionError = Exception
    mock_redis.RedisError = Exception
    return mock_redis


class TestRedisCacheBackend:
    """Tests for Redis cache backend."""

    @patch("mcp_hub.redis_cache.REDIS_AVAILABLE", True)
    @patch("mcp_hub.redis_cache.redis")
    def test_redis_backend_initialization(self, mock_redis_module):
        """Test Redis backend initialization."""
        from mcp_hub.redis_cache import RedisCacheBackend

        mock_redis_module.Redis.return_value = MockRedis()
        mock_redis_module.ConnectionPool = MockConnectionPool

        backend = RedisCacheBackend(
            host="localhost", port=6379, db=0, default_ttl=3600
        )

        assert backend.default_ttl == 3600
        assert backend.key_prefix == "mcp_hub:"

    @patch("mcp_hub.redis_cache.REDIS_AVAILABLE", True)
    @patch("mcp_hub.redis_cache.redis")
    def test_redis_backend_initialization_with_url(self, mock_redis_module):
        """Test Redis backend initialization with URL."""
        from mcp_hub.redis_cache import RedisCacheBackend

        mock_redis_module.Redis.return_value = MockRedis()
        mock_redis_module.ConnectionPool = MockConnectionPool

        backend = RedisCacheBackend(url="redis://localhost:6379/0")

        assert backend is not None

    @patch("mcp_hub.redis_cache.REDIS_AVAILABLE", False)
    def test_redis_backend_not_available(self):
        """Test error when Redis is not installed."""
        from mcp_hub.redis_cache import RedisCacheBackend

        with pytest.raises(RuntimeError, match="Redis is not installed"):
            RedisCacheBackend()

    @patch("mcp_hub.redis_cache.REDIS_AVAILABLE", True)
    @patch("mcp_hub.redis_cache.redis")
    def test_get_and_set(self, mock_redis_module):
        """Test getting and setting cache values."""
        from mcp_hub.redis_cache import RedisCacheBackend

        mock_client = MockRedis()
        mock_redis_module.Redis.return_value = mock_client
        mock_redis_module.ConnectionPool = MockConnectionPool

        backend = RedisCacheBackend()

        # Set a value
        backend.set("test_key", "test_value", ttl=60)

        # Get the value
        result = backend.get("test_key")
        assert result == "test_value"

    @patch("mcp_hub.redis_cache.REDIS_AVAILABLE", True)
    @patch("mcp_hub.redis_cache.redis")
    def test_cache_miss(self, mock_redis_module):
        """Test cache miss returns None."""
        from mcp_hub.redis_cache import RedisCacheBackend

        mock_redis_module.Redis.return_value = MockRedis()
        mock_redis_module.ConnectionPool = MockConnectionPool

        backend = RedisCacheBackend()

        result = backend.get("nonexistent_key")
        assert result is None

    @patch("mcp_hub.redis_cache.REDIS_AVAILABLE", True)
    @patch("mcp_hub.redis_cache.redis")
    def test_delete(self, mock_redis_module):
        """Test deleting cache entries."""
        from mcp_hub.redis_cache import RedisCacheBackend

        mock_client = MockRedis()
        mock_redis_module.Redis.return_value = mock_client
        mock_redis_module.ConnectionPool = MockConnectionPool

        backend = RedisCacheBackend()

        # Set and delete
        backend.set("test_key", "test_value")
        result = backend.delete("test_key")
        assert result is True

        # Verify deleted
        assert backend.get("test_key") is None

    @patch("mcp_hub.redis_cache.REDIS_AVAILABLE", True)
    @patch("mcp_hub.redis_cache.redis")
    def test_cached_call(self, mock_redis_module):
        """Test cached function calls."""
        from mcp_hub.redis_cache import RedisCacheBackend

        mock_client = MockRedis()
        mock_redis_module.Redis.return_value = mock_client
        mock_redis_module.ConnectionPool = MockConnectionPool

        backend = RedisCacheBackend()

        # Mock function
        mock_func = Mock(return_value="result")
        mock_func.__name__ = "test_func"

        # First call - should execute function
        result1 = backend.cached_call(mock_func, (), {})
        assert result1 == "result"
        assert mock_func.call_count == 1

        # Second call - should use cache
        result2 = backend.cached_call(mock_func, (), {})
        assert result2 == "result"
        assert mock_func.call_count == 1  # Not called again

    @patch("mcp_hub.redis_cache.REDIS_AVAILABLE", True)
    @patch("mcp_hub.redis_cache.redis")
    def test_clear_all(self, mock_redis_module):
        """Test clearing all cache entries."""
        from mcp_hub.redis_cache import RedisCacheBackend

        mock_client = MockRedis()
        mock_redis_module.Redis.return_value = mock_client
        mock_redis_module.ConnectionPool = MockConnectionPool

        backend = RedisCacheBackend()

        # Set multiple values
        backend.set("key1", "value1")
        backend.set("key2", "value2")
        backend.set("key3", "value3")

        # Clear all
        count = backend.clear_all()
        assert count == 3

        # Verify cleared
        assert backend.get("key1") is None
        assert backend.get("key2") is None

    @patch("mcp_hub.redis_cache.REDIS_AVAILABLE", True)
    @patch("mcp_hub.redis_cache.redis")
    def test_get_cache_status(self, mock_redis_module):
        """Test getting cache status."""
        from mcp_hub.redis_cache import RedisCacheBackend

        mock_client = MockRedis()
        mock_redis_module.Redis.return_value = mock_client
        mock_redis_module.ConnectionPool = MockConnectionPool

        backend = RedisCacheBackend()

        # Set some values
        backend.set("key1", "value1")
        backend.set("key2", "value2")

        status = backend.get_cache_status()

        assert status["status"] == "healthy"
        assert status["backend"] == "redis"
        assert "total_keys" in status
        assert "redis_version" in status

    @patch("mcp_hub.redis_cache.REDIS_AVAILABLE", True)
    @patch("mcp_hub.redis_cache.redis")
    def test_health_check(self, mock_redis_module):
        """Test health check."""
        from mcp_hub.redis_cache import RedisCacheBackend

        mock_client = MockRedis()
        mock_redis_module.Redis.return_value = mock_client
        mock_redis_module.ConnectionPool = MockConnectionPool

        backend = RedisCacheBackend()

        assert backend.health_check() is True


class TestCacheFactory:
    """Tests for cache factory function."""

    @patch("mcp_hub.cache_utils.cache_config")
    def test_create_file_cache_manager(self, mock_config):
        """Test creating file-based cache manager."""
        from mcp_hub.cache_utils import create_cache_manager

        mock_config.cache_backend = "file"
        mock_config.cache_dir = "test_cache"
        mock_config.default_ttl = 1800

        manager = create_cache_manager()

        from mcp_hub.cache_utils import CacheManager

        assert isinstance(manager, CacheManager)

    @patch("mcp_hub.cache_utils.cache_config")
    @patch("mcp_hub.redis_cache.REDIS_AVAILABLE", True)
    @patch("mcp_hub.redis_cache.redis")
    def test_create_redis_cache_manager(self, mock_redis, mock_config):
        """Test creating Redis cache manager."""
        from mcp_hub.cache_utils import create_cache_manager

        mock_config.cache_backend = "redis"
        mock_config.redis_url = ""
        mock_config.redis_host = "localhost"
        mock_config.redis_port = 6379
        mock_config.redis_db = 0
        mock_config.redis_password = ""
        mock_config.redis_ssl = False
        mock_config.redis_socket_timeout = 5
        mock_config.redis_socket_connect_timeout = 5
        mock_config.redis_max_connections = 50
        mock_config.default_ttl = 3600

        mock_redis.Redis.return_value = MockRedis()
        mock_redis.ConnectionPool = MockConnectionPool

        manager = create_cache_manager()

        from mcp_hub.redis_cache import RedisCacheBackend

        assert isinstance(manager, RedisCacheBackend)

    @patch("mcp_hub.cache_utils.cache_config")
    def test_create_cache_manager_fallback(self, mock_config):
        """Test fallback to file cache when Redis fails."""
        from mcp_hub.cache_utils import create_cache_manager

        mock_config.cache_backend = "redis"
        mock_config.cache_dir = "test_cache"
        mock_config.default_ttl = 1800

        # Redis will fail to import/initialize
        manager = create_cache_manager()

        from mcp_hub.cache_utils import CacheManager

        # Should fall back to file cache
        assert isinstance(manager, CacheManager)


class TestRedisIntegration:
    """Integration tests for Redis caching."""

    @patch("mcp_hub.redis_cache.REDIS_AVAILABLE", True)
    @patch("mcp_hub.redis_cache.redis")
    def test_cache_with_complex_objects(self, mock_redis_module):
        """Test caching complex Python objects."""
        from mcp_hub.redis_cache import RedisCacheBackend

        mock_client = MockRedis()
        mock_redis_module.Redis.return_value = mock_client
        mock_redis_module.ConnectionPool = MockConnectionPool

        backend = RedisCacheBackend()

        # Cache complex object
        complex_obj = {
            "list": [1, 2, 3],
            "dict": {"key": "value"},
            "nested": {"deep": {"data": "here"}},
        }

        backend.set("complex", complex_obj)
        result = backend.get("complex")

        assert result == complex_obj

    @patch("mcp_hub.redis_cache.REDIS_AVAILABLE", True)
    @patch("mcp_hub.redis_cache.redis")
    def test_key_prefix_isolation(self, mock_redis_module):
        """Test that key prefix isolates cache entries."""
        from mcp_hub.redis_cache import RedisCacheBackend

        mock_client = MockRedis()
        mock_redis_module.Redis.return_value = mock_client
        mock_redis_module.ConnectionPool = MockConnectionPool

        backend = RedisCacheBackend(key_prefix="test:")

        backend.set("key1", "value1")

        # The actual Redis key should have the prefix
        prefixed_key = backend._make_key("key1")
        assert prefixed_key.startswith("test:")
