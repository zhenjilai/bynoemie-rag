"""
LLM Module for ByNoemie RAG Chatbot

Provides unified interface for multiple LLM providers:
- Groq (FREE tier, recommended)
- OpenAI (GPT-4, GPT-4o-mini)
- Anthropic (Claude 3)
- Ollama (Local, FREE)

Usage:
    from src.llm import create_llm_client, LLMClientFactory
    
    # Create specific client
    client = LLMClientFactory.create("groq", api_key="...")
    
    # Create with fallback
    client = create_llm_client()
    
    # Generate response
    response = client.chat(
        system_prompt="You are a helpful assistant",
        user_prompt="Hello!"
    )
    print(response.content)
    
    # Get LangChain model for use with LangGraph
    langchain_model = client.get_langchain_model()
"""

from .base import (
    BaseLLMClient,
    LLMResponse,
    LLMClientFactory,
    FallbackLLMClient
)

from .groq_client import GroqClient
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .ollama_client import OllamaClient

from .utils import (
    parse_json_response,
    count_tokens,
    truncate_to_tokens,
    format_messages,
    extract_code_blocks,
    clean_llm_output,
    validate_vibe_tags
)


def create_llm_client(
    provider: str = None,
    **kwargs
) -> BaseLLMClient:
    """
    Create an LLM client with automatic fallback.
    
    Args:
        provider: Specific provider to use (optional)
        **kwargs: Provider-specific configuration
    
    Returns:
        Configured LLM client
    """
    import os
    
    # Load settings if available
    try:
        from config import settings
        
        if provider is None:
            provider = settings.llm.primary_provider
        
        config = settings.llm.providers.get(provider, {})
        config.update(kwargs)
        
    except ImportError:
        config = kwargs
    
    # If specific provider requested
    if provider:
        return LLMClientFactory.create(provider, **config)
    
    # Try providers in order
    providers = ["groq", "openai", "anthropic", "ollama"]
    
    for prov in providers:
        try:
            client = LLMClientFactory.create(prov, **config)
            if client.is_available():
                return client
        except Exception:
            continue
    
    raise RuntimeError("No LLM providers available")


def create_fallback_client(
    providers: list = None,
    configs: dict = None
) -> FallbackLLMClient:
    """
    Create a client with automatic fallback between providers.
    
    Args:
        providers: List of provider names in priority order
        configs: Dict of provider-specific configurations
    
    Returns:
        FallbackLLMClient that tries each provider
    """
    if providers is None:
        providers = ["groq", "openai", "ollama"]
    
    if configs is None:
        configs = {}
    
    return FallbackLLMClient(providers=providers, configs=configs)


__all__ = [
    # Base classes
    "BaseLLMClient",
    "LLMResponse", 
    "LLMClientFactory",
    "FallbackLLMClient",
    
    # Clients
    "GroqClient",
    "OpenAIClient",
    "AnthropicClient",
    "OllamaClient",
    
    # Factory functions
    "create_llm_client",
    "create_fallback_client",
    
    # Utilities
    "parse_json_response",
    "count_tokens",
    "truncate_to_tokens",
    "format_messages",
    "extract_code_blocks",
    "clean_llm_output",
    "validate_vibe_tags",
]
