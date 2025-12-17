"""
Rate Limiter Utility

Implements rate limiting for API calls to respect provider limits.
"""

import time
import logging
import threading
from typing import Dict, Optional
from dataclasses import dataclass, field
from collections import deque
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    requests_per_day: int = 0  # 0 = unlimited
    retry_after_seconds: float = 1.0
    max_retries: int = 3


class RateLimiter:
    """
    Token bucket rate limiter with sliding window.
    
    Features:
    - Tracks both requests and tokens
    - Thread-safe
    - Automatic retry with backoff
    """
    
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self._request_times: deque = deque()
        self._token_usage: deque = deque()
        self._daily_requests: int = 0
        self._daily_reset_time: float = time.time()
        self._lock = threading.Lock()
    
    def _clean_old_entries(self):
        """Remove entries older than 1 minute"""
        now = time.time()
        cutoff = now - 60
        
        while self._request_times and self._request_times[0] < cutoff:
            self._request_times.popleft()
        
        while self._token_usage and self._token_usage[0][0] < cutoff:
            self._token_usage.popleft()
        
        # Reset daily counter if needed
        if now - self._daily_reset_time > 86400:
            self._daily_requests = 0
            self._daily_reset_time = now
    
    def check_limit(self, estimated_tokens: int = 0) -> tuple[bool, float]:
        """
        Check if request is within rate limits.
        
        Args:
            estimated_tokens: Estimated tokens for the request
            
        Returns:
            (allowed, wait_time): Whether request is allowed and wait time if not
        """
        with self._lock:
            self._clean_old_entries()
            
            now = time.time()
            
            # Check requests per minute
            if len(self._request_times) >= self.config.requests_per_minute:
                wait_time = 60 - (now - self._request_times[0])
                return False, max(0, wait_time)
            
            # Check tokens per minute
            total_tokens = sum(t[1] for t in self._token_usage) + estimated_tokens
            if total_tokens > self.config.tokens_per_minute:
                wait_time = 60 - (now - self._token_usage[0][0])
                return False, max(0, wait_time)
            
            # Check daily limit
            if self.config.requests_per_day > 0:
                if self._daily_requests >= self.config.requests_per_day:
                    wait_time = 86400 - (now - self._daily_reset_time)
                    return False, wait_time
            
            return True, 0
    
    def record_request(self, tokens_used: int = 0):
        """Record a completed request"""
        with self._lock:
            now = time.time()
            self._request_times.append(now)
            if tokens_used > 0:
                self._token_usage.append((now, tokens_used))
            self._daily_requests += 1
    
    def wait_if_needed(self, estimated_tokens: int = 0) -> float:
        """
        Wait if rate limit would be exceeded.
        
        Returns:
            Time waited in seconds
        """
        allowed, wait_time = self.check_limit(estimated_tokens)
        
        if not allowed and wait_time > 0:
            logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
            time.sleep(wait_time)
            return wait_time
        
        return 0
    
    def get_status(self) -> Dict:
        """Get current rate limit status"""
        with self._lock:
            self._clean_old_entries()
            
            return {
                "requests_in_window": len(self._request_times),
                "requests_limit": self.config.requests_per_minute,
                "tokens_in_window": sum(t[1] for t in self._token_usage),
                "tokens_limit": self.config.tokens_per_minute,
                "daily_requests": self._daily_requests,
                "daily_limit": self.config.requests_per_day
            }


class MultiProviderRateLimiter:
    """Rate limiter that tracks multiple providers"""
    
    def __init__(self):
        self._limiters: Dict[str, RateLimiter] = {}
    
    def get_limiter(self, provider: str) -> RateLimiter:
        """Get or create limiter for provider"""
        if provider not in self._limiters:
            # Load config from settings
            try:
                from config import settings
                limits = settings.get_rate_limit(provider)
                config = RateLimitConfig(**limits)
            except:
                config = RateLimitConfig()
            
            self._limiters[provider] = RateLimiter(config)
        
        return self._limiters[provider]
    
    def check_and_wait(self, provider: str, estimated_tokens: int = 0) -> float:
        """Check limit and wait if needed for provider"""
        limiter = self.get_limiter(provider)
        return limiter.wait_if_needed(estimated_tokens)
    
    def record(self, provider: str, tokens_used: int = 0):
        """Record request for provider"""
        limiter = self.get_limiter(provider)
        limiter.record_request(tokens_used)


def rate_limit(provider: str = "default", estimated_tokens: int = 100):
    """
    Decorator for rate-limited functions.
    
    Usage:
        @rate_limit(provider="groq", estimated_tokens=500)
        def call_api():
            ...
    """
    def decorator(func):
        limiter = RateLimiter()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter.wait_if_needed(estimated_tokens)
            result = func(*args, **kwargs)
            limiter.record_request(estimated_tokens)
            return result
        
        return wrapper
    return decorator


# Singleton
_rate_limiter: Optional[MultiProviderRateLimiter] = None


def get_rate_limiter() -> MultiProviderRateLimiter:
    """Get singleton rate limiter"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = MultiProviderRateLimiter()
    return _rate_limiter
