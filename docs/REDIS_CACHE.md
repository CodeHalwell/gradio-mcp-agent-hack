# Redis Caching for Distributed Deployments

The MCP Hub supports Redis-based caching for distributed deployments where multiple instances need to share cached data. This is essential for load-balanced setups, Kubernetes clusters, and multi-instance production environments.

## Overview

The caching system provides:
- **Dual Backends**: File-based (single instance) or Redis-based (distributed)
- **Transparent Interface**: Same API regardless of backend
- **Automatic Fallback**: Falls back to file cache if Redis unavailable
- **Connection Pooling**: Efficient connection management
- **TTL Support**: Configurable time-to-live for cache entries
- **Health Monitoring**: Built-in health checks and status reporting

## Why Redis for Distributed Caching?

### File-Based Cache Limitations

- ✗ Each instance has its own cache (no sharing)
- ✗ Wasted memory with duplicate cache entries
- ✗ Slower cold starts for new instances
- ✗ No cache invalidation across instances

### Redis Cache Benefits

- ✓ Shared cache across all instances
- ✓ Reduced API calls (shared benefit)
- ✓ Faster warm-up for new instances
- ✓ Centralized cache management
- ✓ Better for horizontal scaling

## Installation

### Install Redis Dependency

```bash
pip install redis>=5.0.0
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

### Install Redis Server

#### Using Docker

```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

#### Using Docker Compose

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes

  mcp-hub:
    build: .
    ports:
      - "7860:7860"
    environment:
      - CACHE_BACKEND=redis
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      - redis

volumes:
  redis-data:
```

#### Native Installation

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install redis-server
sudo systemctl start redis-server
```

**MacOS:**
```bash
brew install redis
brew services start redis
```

**Verify Installation:**
```bash
redis-cli ping
# Should return: PONG
```

## Configuration

### Environment Variables

Configure caching via environment variables in your `.env` file:

```bash
# Cache Backend Selection
CACHE_BACKEND=redis  # Options: "file" or "redis"

# File Cache Settings (used when CACHE_BACKEND=file)
CACHE_DIR=cache
CACHE_DEFAULT_TTL=3600

# Redis Connection Settings
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=  # Optional: leave empty if no password

# Advanced Redis Settings
REDIS_SSL=false  # Set to "true" for SSL connections
REDIS_MAX_CONNECTIONS=50

# Or use a Redis URL (takes precedence over individual settings)
REDIS_URL=redis://localhost:6379/0
# For password-protected Redis:
# REDIS_URL=redis://:password@localhost:6379/0
# For Redis with SSL:
# REDIS_URL=rediss://localhost:6379/0
```

### Configuration Examples

#### Local Development (File Cache)

```bash
CACHE_BACKEND=file
CACHE_DIR=cache
CACHE_DEFAULT_TTL=3600
```

#### Production (Redis)

```bash
CACHE_BACKEND=redis
REDIS_HOST=redis.example.com
REDIS_PORT=6379
REDIS_PASSWORD=your_secure_password
REDIS_SSL=true
```

#### Kubernetes (Redis Service)

```bash
CACHE_BACKEND=redis
REDIS_HOST=redis-service
REDIS_PORT=6379
REDIS_DB=0
```

#### AWS ElastiCache

```bash
CACHE_BACKEND=redis
REDIS_URL=rediss://master.my-cache.abc123.use1.cache.amazonaws.com:6379
REDIS_SSL=true
```

#### Azure Cache for Redis

```bash
CACHE_BACKEND=redis
REDIS_HOST=myredis.redis.cache.windows.net
REDIS_PORT=6380
REDIS_PASSWORD=your_access_key
REDIS_SSL=true
```

## Usage

### Transparent Caching

The cache backend is selected automatically based on `CACHE_BACKEND`. Your code doesn't need to change:

```python
from mcp_hub.cache_utils import cache_manager

# Set a value (works with both backends)
cache_manager.set("my_key", {"data": "value"}, ttl=600)

# Get a value
result = cache_manager.get("my_key")

# Delete a value
cache_manager.delete("my_key")

# Clear all cache
cache_manager.clear_all()
```

### Using the Decorator

```python
from mcp_hub.cache_utils import cached

@cached(ttl=1800)  # Cache for 30 minutes
def expensive_operation(param1, param2):
    # This will be cached automatically
    result = perform_expensive_computation(param1, param2)
    return result

# First call - executes function and caches result
result1 = expensive_operation("a", "b")

# Second call with same args - returns cached result
result2 = expensive_operation("a", "b")
```

### Cache Status Monitoring

```python
status = cache_manager.get_cache_status()
print(status)
```

#### File Cache Status:
```json
{
  "status": "healthy",
  "cache_dir": "cache",
  "total_files": 150,
  "expired_files": 5,
  "total_size_mb": 45.2,
  "default_ttl_seconds": 3600
}
```

#### Redis Cache Status:
```json
{
  "status": "healthy",
  "backend": "redis",
  "redis_version": "7.0.0",
  "connected_clients": 10,
  "total_keys": 250,
  "memory_used_human": "15M",
  "default_ttl_seconds": 3600,
  "key_prefix": "mcp_hub:"
}
```

## Kubernetes Deployment

### Redis StatefulSet

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
spec:
  serviceName: redis
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
          name: redis
        volumeMounts:
        - name: redis-data
          mountPath: /data
  volumeClaimTemplates:
  - metadata:
      name: redis-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
  clusterIP: None
```

### MCP Hub Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-hub
spec:
  replicas: 3  # Multiple instances sharing Redis cache
  selector:
    matchLabels:
      app: mcp-hub
  template:
    metadata:
      labels:
        app: mcp-hub
    spec:
      containers:
      - name: mcp-hub
        image: mcp-hub:latest
        env:
        - name: CACHE_BACKEND
          value: "redis"
        - name: REDIS_HOST
          value: "redis-service"
        - name: REDIS_PORT
          value: "6379"
        - name: CACHE_DEFAULT_TTL
          value: "3600"
        ports:
        - containerPort: 7860
```

## Performance Considerations

### TTL Configuration

Choose appropriate TTL based on data freshness needs:

```bash
# Short TTL for frequently changing data
CACHE_DEFAULT_TTL=300  # 5 minutes

# Medium TTL for semi-static data
CACHE_DEFAULT_TTL=1800  # 30 minutes

# Long TTL for rarely changing data
CACHE_DEFAULT_TTL=3600  # 1 hour
```

### Connection Pooling

Redis uses connection pooling for efficiency:

```bash
# Adjust based on concurrent request load
REDIS_MAX_CONNECTIONS=50  # Default
REDIS_MAX_CONNECTIONS=100  # High traffic
```

### Memory Management

Monitor Redis memory usage:

```bash
# Connect to Redis
redis-cli

# Check memory usage
> INFO memory

# Check keys
> DBSIZE

# View cache keys
> KEYS mcp_hub:*
```

### Cache Warming

For production deployments, pre-warm the cache:

```python
# Warm common searches
common_queries = ["python list", "async await", "docker basics"]
for query in common_queries:
    web_search_agent.search(query)
```

## Troubleshooting

### Redis Connection Failed

**Error:** `Redis connection failed: Error connecting to redis:6379`

**Solutions:**
1. Check Redis is running: `redis-cli ping`
2. Verify network connectivity: `telnet redis-host 6379`
3. Check firewall rules allow port 6379
4. Verify REDIS_HOST is correct
5. Check Redis logs: `docker logs redis`

### Fallback to File Cache

If Redis fails, the system automatically falls back to file cache:

```
ERROR Failed to initialize Redis cache backend: [Error]
INFO Falling back to file-based cache
```

To fix:
1. Check Redis server is running
2. Verify connection settings
3. Check Redis logs for errors

### Authentication Failed

**Error:** `NOAUTH Authentication required`

**Solution:** Set Redis password in config:
```bash
REDIS_PASSWORD=your_password
# Or use URL
REDIS_URL=redis://:your_password@localhost:6379/0
```

### SSL/TLS Errors

**Error:** `SSL: CERTIFICATE_VERIFY_FAILED`

**Solution:** Enable SSL in config:
```bash
REDIS_SSL=true
REDIS_URL=rediss://your-host:6380
```

### High Memory Usage

Monitor and manage Redis memory:

```bash
# Set max memory limit (in Redis config)
maxmemory 2gb
maxmemory-policy allkeys-lru

# Or via command
redis-cli CONFIG SET maxmemory 2gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

## Monitoring

### Health Check

```python
from mcp_hub.cache_utils import cache_manager

# Check cache health
if hasattr(cache_manager, 'health_check'):
    is_healthy = cache_manager.health_check()
    print(f"Cache is {'healthy' if is_healthy else 'unhealthy'}")
```

### Redis Monitoring Tools

**redis-cli:**
```bash
redis-cli --stat
# Shows real-time stats
```

**RedisInsight:**
- GUI tool for Redis monitoring
- Download: https://redis.com/redis-enterprise/redis-insight/

**Prometheus Exporter:**
```yaml
# Add redis_exporter to Docker Compose
  redis-exporter:
    image: oliver006/redis_exporter
    ports:
      - "9121:9121"
    environment:
      - REDIS_ADDR=redis:6379
```

## Best Practices

### 1. Use Redis in Production

Always use Redis for production deployments with multiple instances:

```bash
# Production
CACHE_BACKEND=redis

# Local development
CACHE_BACKEND=file
```

### 2. Set Appropriate TTLs

Don't cache forever - set reasonable TTLs:

```python
# Good: Reasonable TTL
@cached(ttl=1800)  # 30 minutes

# Bad: Too long or infinite
@cached(ttl=86400)  # 24 hours - may serve stale data
```

### 3. Use Connection Pooling

Let the Redis client manage connections:

```bash
# Don't set too high
REDIS_MAX_CONNECTIONS=50  # Good for most deployments

# Adjust based on load
REDIS_MAX_CONNECTIONS=100  # High traffic applications
```

### 4. Monitor Cache Hit Rate

Track cache effectiveness:

```python
# Prometheus metrics automatically track this
mcp_hub_cache_hits_total
mcp_hub_cache_misses_total

# Calculate hit rate
hit_rate = hits / (hits + misses)
```

### 5. Handle Failures Gracefully

The system automatically falls back to file cache if Redis fails. Ensure this behavior is acceptable for your use case.

### 6. Secure Redis

For production:

```bash
# Use password
REDIS_PASSWORD=strong_random_password

# Use SSL
REDIS_SSL=true

# Bind to specific IP (in redis.conf)
bind 127.0.0.1

# Use firewall rules to restrict access
```

### 7. Regular Maintenance

```bash
# Flush expired keys
redis-cli --scan --pattern "mcp_hub:*" | xargs redis-cli del

# Backup Redis data
redis-cli BGSAVE

# Monitor memory
redis-cli INFO memory
```

## Migration

### From File Cache to Redis

1. Install Redis and redis Python package
2. Update environment variables:
   ```bash
   CACHE_BACKEND=redis
   REDIS_HOST=your-redis-host
   ```
3. Restart application
4. Cache will rebuild automatically

### From Redis to File Cache

1. Update environment variables:
   ```bash
   CACHE_BACKEND=file
   ```
2. Restart application
3. Remove Redis service (optional)

## Architecture

```
┌─────────────────────────────────────────────────┐
│           Application Code                       │
│  @cached decorator / cache_manager.get/set       │
└───────────────────┬─────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  cache_utils.py       │
        │  create_cache_manager()│
        └───────────┬───────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
         ▼                     ▼
┌─────────────────┐   ┌────────────────────┐
│  CacheManager   │   │ RedisCacheBackend  │
│  (File-based)   │   │  (Redis-based)     │
└────────┬────────┘   └─────────┬──────────┘
         │                       │
         ▼                       ▼
   ┌──────────┐          ┌────────────┐
   │ Disk     │          │ Redis      │
   │ Files    │          │ Server     │
   └──────────┘          └────────────┘
```

## Further Reading

- [Redis Documentation](https://redis.io/docs/)
- [Redis Best Practices](https://redis.io/docs/management/optimization/)
- [Redis Python Client](https://redis-py.readthedocs.io/)
- [Caching Strategies](https://docs.aws.amazon.com/AmazonElastiCache/latest/red-ug/Strategies.html)
