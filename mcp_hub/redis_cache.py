"""Redis-based caching backend for distributed deployments.

This module provides a Redis cache implementation that can be used
as a drop-in replacement for the file-based cache in distributed
environments where multiple instances need to share cache data.
"""

import hashlib
import json
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from .logging_config import logger

try:
    import redis
    from redis import ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not installed. Install with: pip install redis>=5.0.0")


class RedisCacheBackend:
    """Redis-based cache backend for distributed caching.

    This backend uses Redis for caching, which allows multiple instances
    of the application to share the same cache. This is essential for
    distributed deployments, load-balanced setups, or Kubernetes clusters.

    Features:
    - Connection pooling for better performance
    - Automatic serialization/deserialization with pickle
    - TTL (time-to-live) support
    - Graceful error handling and fallback
    - Health checking and monitoring
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ssl: bool = False,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        max_connections: int = 50,
        url: Optional[str] = None,
        default_ttl: int = 3600,
        key_prefix: str = "mcp_hub:",
    ):
        """
        Initialize Redis cache backend.

        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            password: Redis password (if required)
            ssl: Use SSL connection
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Socket connect timeout in seconds
            max_connections: Maximum connections in pool
            url: Redis connection URL (takes precedence over individual params)
            default_ttl: Default time-to-live in seconds
            key_prefix: Prefix for all cache keys
        """
        if not REDIS_AVAILABLE:
            raise RuntimeError(
                "Redis is not installed. Install with: pip install redis>=5.0.0"
            )

        self.default_ttl = default_ttl
        self.key_prefix = key_prefix

        try:
            if url:
                # Use connection URL
                self.pool = ConnectionPool.from_url(
                    url,
                    max_connections=max_connections,
                    socket_timeout=socket_timeout,
                    socket_connect_timeout=socket_connect_timeout,
                )
            else:
                # Use individual parameters
                self.pool = ConnectionPool(
                    host=host,
                    port=port,
                    db=db,
                    password=password if password else None,
                    ssl=ssl,
                    max_connections=max_connections,
                    socket_timeout=socket_timeout,
                    socket_connect_timeout=socket_connect_timeout,
                )

            self.client = redis.Redis(connection_pool=self.pool)

            # Test connection
            self.client.ping()
            logger.info(f"Redis cache backend initialized: {host}:{port} db={db}")

        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise RuntimeError(f"Redis connection failed: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            raise

    def _make_key(self, cache_key: str) -> str:
        """Add prefix to cache key.

        Args:
            cache_key: The cache key

        Returns:
            Prefixed cache key
        """
        return f"{self.key_prefix}{cache_key}"

    def _get_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a unique cache key based on function name and arguments.

        Args:
            func_name: Name of the function
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            MD5 hash of the key data
        """
        key_data = {"func": func_name, "args": args, "kwargs": kwargs}
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, cache_key: str) -> Optional[Any]:
        """Retrieve a value from cache.

        Args:
            cache_key: The cache key to retrieve

        Returns:
            Cached value if found and not expired, None otherwise
        """
        try:
            key = self._make_key(cache_key)
            data = self.client.get(key)

            if data is None:
                logger.debug(f"Cache miss for key: {cache_key}")
                return None

            # Deserialize the cached data
            cached_value = pickle.loads(data)
            logger.debug(f"Cache hit for key: {cache_key}")
            return cached_value

        except redis.RedisError as e:
            logger.error(f"Redis error getting key {cache_key}: {e}")
            return None
        except pickle.PickleError as e:
            logger.error(f"Failed to deserialize cached value for {cache_key}: {e}")
            # Delete corrupted cache entry
            try:
                self.client.delete(self._make_key(cache_key))
            except Exception:
                pass
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting cache key {cache_key}: {e}")
            return None

    def set(self, cache_key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store a value in cache.

        Args:
            cache_key: The cache key
            value: The value to cache
            ttl: Time-to-live in seconds (None for default)

        Returns:
            True if successful, False otherwise
        """
        if ttl is None:
            ttl = self.default_ttl

        try:
            key = self._make_key(cache_key)
            serialized_value = pickle.dumps(value)

            # Set with TTL
            self.client.setex(key, ttl, serialized_value)
            logger.debug(f"Cached value for key: {cache_key} (TTL: {ttl}s)")
            return True

        except redis.RedisError as e:
            logger.error(f"Redis error setting key {cache_key}: {e}")
            return False
        except pickle.PickleError as e:
            logger.error(f"Failed to serialize value for {cache_key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error setting cache key {cache_key}: {e}")
            return False

    def delete(self, cache_key: str) -> bool:
        """Delete a cache entry.

        Args:
            cache_key: The cache key to delete

        Returns:
            True if deleted, False otherwise
        """
        try:
            key = self._make_key(cache_key)
            result = self.client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Error deleting cache key {cache_key}: {e}")
            return False

    def cached_call(
        self, func, args: tuple, kwargs: dict, ttl: Optional[int] = None
    ) -> Any:
        """Make a cached function call.

        Args:
            func: Function to call
            args: Positional arguments
            kwargs: Keyword arguments
            ttl: Time-to-live in seconds

        Returns:
            Function result (from cache or fresh execution)
        """
        cache_key = self._get_cache_key(func.__name__, args, kwargs)

        # Try to get from cache first
        cached_result = self.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Execute function and cache result
        logger.debug(f"Cache miss for {func.__name__}, executing function")
        result = func(*args, **kwargs)
        self.set(cache_key, result, ttl)

        return result

    def clear_all(self) -> int:
        """Clear all cache entries with the configured prefix.

        Returns:
            Number of keys deleted
        """
        try:
            pattern = f"{self.key_prefix}*"
            keys = list(self.client.scan_iter(match=pattern, count=100))

            if not keys:
                return 0

            deleted = self.client.delete(*keys)
            logger.info(f"Cleared {deleted} cache entries")
            return deleted

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0

    def get_cache_status(self) -> Dict[str, Any]:
        """Get detailed status information about the cache.

        Returns:
            Dictionary with cache status information
        """
        try:
            # Get Redis info
            info = self.client.info()

            # Count keys with our prefix
            pattern = f"{self.key_prefix}*"
            cursor = 0
            key_count = 0
            while True:
                cursor, keys = self.client.scan(cursor, match=pattern, count=100)
                key_count += len(keys)
                if cursor == 0:
                    break

            # Get memory usage
            memory_used = info.get("used_memory", 0)
            memory_used_human = info.get("used_memory_human", "0B")

            return {
                "status": "healthy",
                "backend": "redis",
                "redis_version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "total_keys": key_count,
                "memory_used_bytes": memory_used,
                "memory_used_human": memory_used_human,
                "default_ttl_seconds": self.default_ttl,
                "key_prefix": self.key_prefix,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get cache status: {e}")
            return {
                "status": "error",
                "backend": "redis",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def health_check(self) -> bool:
        """Check if Redis connection is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            self.client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    def close(self):
        """Close the Redis connection pool."""
        try:
            self.pool.disconnect()
            logger.info("Redis connection pool closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")
