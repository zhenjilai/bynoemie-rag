"""
Cache Utility

Provides caching for LLM responses and embeddings.
"""

import os
import json
import hashlib
import time
import logging
from typing import Any, Optional, Dict
from pathlib import Path
from functools import wraps
from dataclasses import dataclass
import pickle

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached entry with metadata"""
    value: Any
    created_at: float
    expires_at: float
    hits: int = 0
    

class BaseCache:
    """Base class for cache implementations"""
    
    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        raise NotImplementedError
    
    def delete(self, key: str) -> bool:
        raise NotImplementedError
    
    def clear(self) -> bool:
        raise NotImplementedError
    
    def exists(self, key: str) -> bool:
        return self.get(key) is not None


class MemoryCache(BaseCache):
    """In-memory cache with TTL support"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        
        # Check expiration
        if time.time() > entry.expires_at:
            del self._cache[key]
            return None
        
        entry.hits += 1
        return entry.value
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        # Evict if at capacity
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        
        ttl = ttl or self.default_ttl
        now = time.time()
        
        self._cache[key] = CacheEntry(
            value=value,
            created_at=now,
            expires_at=now + ttl
        )
        
        return True
    
    def delete(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> bool:
        self._cache.clear()
        return True
    
    def _evict_oldest(self):
        """Evict oldest entries to make room"""
        if not self._cache:
            return
        
        # Sort by created_at, remove oldest 10%
        sorted_keys = sorted(
            self._cache.keys(),
            key=lambda k: self._cache[k].created_at
        )
        
        to_remove = max(1, len(sorted_keys) // 10)
        for key in sorted_keys[:to_remove]:
            del self._cache[key]
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_hits = sum(e.hits for e in self._cache.values())
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "total_hits": total_hits
        }


class DiskCache(BaseCache):
    """Persistent disk-based cache"""
    
    def __init__(
        self,
        directory: str = "./data/cache",
        max_size: int = 1000,
        default_ttl: int = 3600
    ):
        self.directory = Path(directory)
        self.max_size = max_size
        self.default_ttl = default_ttl
        
        # Create directory
        self.directory.mkdir(parents=True, exist_ok=True)
        
        # Index file for metadata
        self.index_file = self.directory / "cache_index.json"
        self._index: Dict[str, Dict] = self._load_index()
    
    def _load_index(self) -> Dict:
        """Load cache index"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_index(self):
        """Save cache index"""
        with open(self.index_file, 'w') as f:
            json.dump(self._index, f)
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for key"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.directory / f"{key_hash}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self._index:
            return None
        
        metadata = self._index[key]
        
        # Check expiration
        if time.time() > metadata["expires_at"]:
            self.delete(key)
            return None
        
        # Load from disk
        file_path = self._get_file_path(key)
        
        if not file_path.exists():
            del self._index[key]
            self._save_index()
            return None
        
        try:
            with open(file_path, 'rb') as f:
                value = pickle.load(f)
            
            # Update hits
            self._index[key]["hits"] = metadata.get("hits", 0) + 1
            self._save_index()
            
            return value
        except Exception as e:
            logger.warning(f"Error loading cache: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        # Evict if at capacity
        if len(self._index) >= self.max_size:
            self._evict_oldest()
        
        ttl = ttl or self.default_ttl
        now = time.time()
        
        # Save to disk
        file_path = self._get_file_path(key)
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Update index
            self._index[key] = {
                "created_at": now,
                "expires_at": now + ttl,
                "hits": 0
            }
            self._save_index()
            
            return True
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        if key not in self._index:
            return False
        
        # Remove file
        file_path = self._get_file_path(key)
        if file_path.exists():
            file_path.unlink()
        
        # Update index
        del self._index[key]
        self._save_index()
        
        return True
    
    def clear(self) -> bool:
        # Remove all cache files
        for file_path in self.directory.glob("*.cache"):
            file_path.unlink()
        
        self._index.clear()
        self._save_index()
        
        return True
    
    def _evict_oldest(self):
        """Evict oldest entries"""
        if not self._index:
            return
        
        sorted_keys = sorted(
            self._index.keys(),
            key=lambda k: self._index[k]["created_at"]
        )
        
        to_remove = max(1, len(sorted_keys) // 10)
        for key in sorted_keys[:to_remove]:
            self.delete(key)


class LLMResponseCache:
    """Specialized cache for LLM responses"""
    
    def __init__(self, cache: BaseCache = None):
        if cache is None:
            try:
                from config import settings
                if settings.cache.backend == "disk":
                    cache = DiskCache(
                        directory=settings.cache.directory,
                        max_size=settings.cache.max_size,
                        default_ttl=settings.cache.ttl_seconds
                    )
                else:
                    cache = MemoryCache(
                        max_size=settings.cache.max_size,
                        default_ttl=settings.cache.ttl_seconds
                    )
            except:
                cache = MemoryCache()
        
        self._cache = cache
    
    def _make_key(
        self,
        model: str,
        messages: list,
        **kwargs
    ) -> str:
        """Create cache key from request parameters"""
        key_data = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get_response(
        self,
        model: str,
        messages: list,
        **kwargs
    ) -> Optional[str]:
        """Get cached response"""
        key = self._make_key(model, messages, **kwargs)
        return self._cache.get(key)
    
    def cache_response(
        self,
        model: str,
        messages: list,
        response: str,
        ttl: int = None,
        **kwargs
    ):
        """Cache a response"""
        key = self._make_key(model, messages, **kwargs)
        self._cache.set(key, response, ttl)


def cached_llm(ttl: int = 3600):
    """
    Decorator to cache LLM function responses.
    
    Usage:
        @cached_llm(ttl=7200)
        def generate_vibes(product_name, description):
            return llm.chat(...)
    """
    cache = LLMResponseCache()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create key from function args
            key_data = {"args": args, "kwargs": kwargs, "func": func.__name__}
            key = hashlib.sha256(
                json.dumps(key_data, sort_keys=True, default=str).encode()
            ).hexdigest()
            
            # Check cache
            cached = cache._cache.get(key)
            if cached is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached
            
            # Execute and cache
            result = func(*args, **kwargs)
            cache._cache.set(key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


# Singleton
_llm_cache: Optional[LLMResponseCache] = None


def get_llm_cache() -> LLMResponseCache:
    """Get singleton LLM cache"""
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = LLMResponseCache()
    return _llm_cache
