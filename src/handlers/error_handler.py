"""
Error Handler Module

Centralized error handling with retry logic and graceful degradation.
"""

import logging
import time
from typing import Optional, Callable, Any, Type, List, Dict
from functools import wraps
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Custom exceptions
class ByNoemieError(Exception):
    """Base exception for ByNoemie chatbot"""
    pass


class LLMError(ByNoemieError):
    """LLM-related errors"""
    pass


class RateLimitError(LLMError):
    """Rate limit exceeded"""
    def __init__(self, message: str, retry_after: float = 60):
        super().__init__(message)
        self.retry_after = retry_after


class TokenLimitError(LLMError):
    """Token limit exceeded"""
    pass


class ProviderError(LLMError):
    """Provider unavailable or returned error"""
    def __init__(self, message: str, provider: str):
        super().__init__(message)
        self.provider = provider


class ValidationError(ByNoemieError):
    """Input validation error"""
    pass


class CacheError(ByNoemieError):
    """Cache-related error"""
    pass


@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    retry_exceptions: List[Type[Exception]] = field(
        default_factory=lambda: [
            RateLimitError,
            ProviderError,
            ConnectionError,
            TimeoutError
        ]
    )


class RetryHandler:
    """
    Handles retries with exponential backoff.
    
    Usage:
        handler = RetryHandler(max_retries=3)
        
        @handler.retry
        def call_api():
            ...
        
        # Or with context manager
        with handler.attempt("api_call"):
            result = call_api()
    """
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self._attempt_count = 0
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        delay = self.config.base_delay * (
            self.config.exponential_base ** attempt
        )
        return min(delay, self.config.max_delay)
    
    def should_retry(self, exception: Exception) -> bool:
        """Check if exception should trigger retry"""
        for exc_type in self.config.retry_exceptions:
            if isinstance(exception, exc_type):
                return True
        return False
    
    def retry(self, func: Callable) -> Callable:
        """Decorator for retry logic"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if not self.should_retry(e):
                        raise
                    
                    if attempt < self.config.max_retries:
                        delay = self.calculate_delay(attempt)
                        
                        # Check for rate limit specific delay
                        if isinstance(e, RateLimitError):
                            delay = max(delay, e.retry_after)
                        
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {self.config.max_retries + 1} attempts failed"
                        )
            
            raise last_exception
        
        return wrapper
    
    def attempt(self, operation: str):
        """Context manager for retry attempts"""
        return _RetryContext(self, operation)


class _RetryContext:
    """Context manager for retry logic"""
    
    def __init__(self, handler: RetryHandler, operation: str):
        self.handler = handler
        self.operation = operation
        self.attempt = 0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is None:
            return False
        
        if not self.handler.should_retry(exc_val):
            return False
        
        self.attempt += 1
        
        if self.attempt <= self.handler.config.max_retries:
            delay = self.handler.calculate_delay(self.attempt - 1)
            
            if isinstance(exc_val, RateLimitError):
                delay = max(delay, exc_val.retry_after)
            
            logger.warning(
                f"{self.operation} attempt {self.attempt} failed. "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
            return True  # Suppress exception, retry
        
        return False  # Re-raise exception


class ErrorHandler:
    """
    Centralized error handler with fallback support.
    
    Usage:
        handler = ErrorHandler()
        
        handler.register_fallback(LLMError, lambda e: "Default response")
        
        result = handler.handle(
            lambda: llm.generate(...),
            fallback="Error occurred"
        )
    """
    
    def __init__(self):
        self._fallbacks: Dict[Type[Exception], Callable] = {}
        self._error_counts: Dict[str, int] = {}
    
    def register_fallback(
        self,
        exception_type: Type[Exception],
        handler: Callable[[Exception], Any]
    ):
        """Register a fallback handler for exception type"""
        self._fallbacks[exception_type] = handler
    
    def handle(
        self,
        operation: Callable,
        fallback: Any = None,
        log_error: bool = True
    ) -> Any:
        """
        Execute operation with error handling.
        
        Args:
            operation: Callable to execute
            fallback: Value to return on error
            log_error: Whether to log errors
            
        Returns:
            Result of operation or fallback value
        """
        try:
            return operation()
        except Exception as e:
            error_type = type(e).__name__
            self._error_counts[error_type] = (
                self._error_counts.get(error_type, 0) + 1
            )
            
            if log_error:
                logger.error(f"Error in operation: {e}")
            
            # Check for registered fallback
            for exc_type, handler in self._fallbacks.items():
                if isinstance(e, exc_type):
                    return handler(e)
            
            # Use provided fallback
            if fallback is not None:
                return fallback
            
            raise
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics"""
        return self._error_counts.copy()


# Decorators for common error patterns
def handle_llm_errors(fallback: Any = None):
    """Decorator to handle LLM errors gracefully"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RateLimitError as e:
                logger.warning(f"Rate limit hit: {e}")
                time.sleep(e.retry_after)
                return func(*args, **kwargs)  # One retry
            except TokenLimitError as e:
                logger.error(f"Token limit exceeded: {e}")
                return fallback
            except LLMError as e:
                logger.error(f"LLM error: {e}")
                return fallback
        return wrapper
    return decorator


def validate_input(**validators):
    """
    Decorator to validate function inputs.
    
    Usage:
        @validate_input(
            product_name=lambda x: len(x) > 0,
            price=lambda x: x > 0
        )
        def process_product(product_name, price):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function parameter names
            import inspect
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            # Combine args and kwargs
            all_args = dict(zip(params, args))
            all_args.update(kwargs)
            
            # Validate
            for param_name, validator in validators.items():
                if param_name in all_args:
                    value = all_args[param_name]
                    if not validator(value):
                        raise ValidationError(
                            f"Invalid value for {param_name}: {value}"
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Singleton instances
_retry_handler: Optional[RetryHandler] = None
_error_handler: Optional[ErrorHandler] = None


def get_retry_handler() -> RetryHandler:
    """Get singleton retry handler"""
    global _retry_handler
    if _retry_handler is None:
        _retry_handler = RetryHandler()
    return _retry_handler


def get_error_handler() -> ErrorHandler:
    """Get singleton error handler"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler
