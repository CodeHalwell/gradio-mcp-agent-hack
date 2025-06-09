"""Caching system for improved performance and reduced API calls."""

import hashlib
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Callable
from functools import wraps
from .logging_config import logger

class CacheManager:
    """Simple file-based cache manager for API responses and computations."""
    
    def __init__(self, cache_dir: str = "cache", default_ttl: int = 3600):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            default_ttl: Default time-to-live in seconds (1 hour default)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.default_ttl = default_ttl
        logger.info(f"Cache manager initialized with directory: {self.cache_dir}")
    
    def _get_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a unique cache key based on function name and arguments."""
        # Create a string representation of arguments
        key_data = {
            "func": func_name,
            "args": args,
            "kwargs": kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{cache_key}.cache"
    
    def get(self, cache_key: str) -> Optional[Any]:
        """Retrieve a value from cache if it exists and is not expired."""
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check if cache has expired
            if datetime.now() > cache_data['expires_at']:
                logger.debug(f"Cache expired for key: {cache_key}")
                cache_path.unlink()  # Delete expired cache
                return None
            
            logger.debug(f"Cache hit for key: {cache_key}")
            return cache_data['value']
            
        except (EOFError, pickle.PickleError, KeyError) as e:
            logger.warning(f"Cache corruption for key {cache_key}: {e}")
            cache_path.unlink()  # Delete corrupted cache
            return None
    
    def set(self, cache_key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a value in cache with optional TTL."""
        if ttl is None:
            ttl = self.default_ttl
        
        cache_data = {
            'value': value,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(seconds=ttl)
        }
        
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.debug(f"Cached value for key: {cache_key} (TTL: {ttl}s)")
        except Exception as e:
            logger.error(f"Failed to cache value for key {cache_key}: {e}")
    
    def cached_call(self, func: Callable, args: tuple, kwargs: dict, ttl: Optional[int] = None) -> Any:
        """Make a cached function call."""
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
    
    def clear_expired(self) -> int:
        """Remove all expired cache files and return count of removed files."""
        removed_count = 0
        current_time = datetime.now()
        
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                if current_time > cache_data['expires_at']:
                    cache_file.unlink()
                    removed_count += 1
                    
            except Exception as e:
                logger.warning(f"Error checking cache file {cache_file}: {e}")
                cache_file.unlink()  # Remove corrupted files
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} expired cache files")
        
        return removed_count
    
    def clear_all(self) -> int:
        """Remove all cache files and return count of removed files."""
        removed_count = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink()
            removed_count += 1
        
        logger.info(f"Cleared all cache: removed {removed_count} files")
        return removed_count
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get detailed status information about the cache system."""
        try:
            # Count cache files
            cache_files = list(self.cache_dir.glob("*.cache"))
            cache_count = len(cache_files)
            
            # Calculate cache directory size
            total_size = sum(f.stat().st_size for f in cache_files)
            
            # Count expired files
            expired_count = 0
            current_time = datetime.now()
            for cache_file in cache_files:
                try:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    if current_time > cache_data['expires_at']:
                        expired_count += 1
                except Exception:
                    expired_count += 1  # Count corrupted files as expired
            
            # Get cache stats
            return {
                "status": "healthy",
                "cache_dir": str(self.cache_dir),
                "total_files": cache_count,
                "expired_files": expired_count,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "default_ttl_seconds": self.default_ttl,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get cache status: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Global cache manager instance
cache_manager = CacheManager()

def cached(ttl: int = 3600):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time-to-live in seconds
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return cache_manager.cached_call(func, args, kwargs, ttl)
        return wrapper
    return decorator

# Specialized caching functions for common operations
@cached(ttl=1800)  # 30 minutes
def cached_web_search(query: str) -> Dict[str, Any]:
    """Cached version of web search - import happens at runtime."""
    # Import at runtime to avoid circular imports
    from tavily import TavilyClient
    client = TavilyClient(api_key="placeholder")  # Will be replaced at runtime
    # This is a placeholder - actual implementation would use the real agent
    return {"query": query, "results": [], "cached": True}

@cached(ttl=3600)  # 1 hour  
def cached_llm_processing(text_input: str, task: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Cached version of LLM processing - import happens at runtime."""
    # This is a placeholder for the caching pattern
    return {"input_text": text_input, "task": task, "cached": True}
