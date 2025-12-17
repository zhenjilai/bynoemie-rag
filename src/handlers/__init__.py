"""
Handlers Module for ByNoemie RAG Chatbot

Provides error handling, retry logic, and graceful degradation.
"""

from .error_handler import (
    # Exceptions
    ByNoemieError,
    LLMError,
    RateLimitError,
    TokenLimitError,
    ProviderError,
    ValidationError,
    CacheError,
    
    # Retry
    RetryConfig,
    RetryHandler,
    
    # Error handling
    ErrorHandler,
    
    # Decorators
    handle_llm_errors,
    validate_input,
    
    # Singletons
    get_retry_handler,
    get_error_handler
)


__all__ = [
    # Exceptions
    "ByNoemieError",
    "LLMError",
    "RateLimitError",
    "TokenLimitError",
    "ProviderError",
    "ValidationError",
    "CacheError",
    
    # Retry
    "RetryConfig",
    "RetryHandler",
    
    # Error handling
    "ErrorHandler",
    
    # Decorators
    "handle_llm_errors",
    "validate_input",
    
    # Singletons
    "get_retry_handler",
    "get_error_handler",
]
