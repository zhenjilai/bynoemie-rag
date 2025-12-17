"""
Base LLM Client Module

Provides abstract base class and factory for LLM providers.
Integrates with LangChain for unified interface.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized LLM response"""
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None
    latency_ms: Optional[float] = None
    raw_response: Optional[Any] = None


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: int = 30,
        **kwargs
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.kwargs = kwargs
        self._client = None
        self._langchain_model = None
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name"""
        pass
    
    @abstractmethod
    def _initialize_client(self):
        """Initialize the underlying client"""
        pass
    
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Generate a response from messages"""
        pass
    
    @abstractmethod
    def get_langchain_model(self) -> BaseChatModel:
        """Get LangChain compatible model"""
        pass
    
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> LLMResponse:
        """Convenience method for simple chat"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self.generate(messages, **kwargs)
    
    def is_available(self) -> bool:
        """Check if the provider is available"""
        try:
            self._initialize_client()
            return True
        except Exception as e:
            logger.warning(f"{self.provider_name} not available: {e}")
            return False


class LLMClientFactory:
    """Factory for creating LLM clients"""
    
    _registry: Dict[str, type] = {}
    
    @classmethod
    def register(cls, provider_name: str):
        """Decorator to register a client class"""
        def wrapper(client_class: type):
            cls._registry[provider_name.lower()] = client_class
            return client_class
        return wrapper
    
    @classmethod
    def create(
        cls,
        provider: str,
        **kwargs
    ) -> BaseLLMClient:
        """Create an LLM client by provider name"""
        provider_lower = provider.lower()
        
        if provider_lower not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown provider: {provider}. Available: {available}"
            )
        
        client_class = cls._registry[provider_lower]
        return client_class(**kwargs)
    
    @classmethod
    def create_with_fallback(
        cls,
        providers: List[str],
        configs: Dict[str, Dict]
    ) -> BaseLLMClient:
        """Create client with fallback chain"""
        for provider in providers:
            try:
                config = configs.get(provider, {})
                client = cls.create(provider, **config)
                if client.is_available():
                    logger.info(f"Using LLM provider: {provider}")
                    return client
            except Exception as e:
                logger.warning(f"Failed to initialize {provider}: {e}")
                continue
        
        raise RuntimeError("No LLM providers available")


class FallbackLLMClient(BaseLLMClient):
    """LLM client with automatic fallback between providers"""
    
    def __init__(
        self,
        providers: List[str],
        configs: Dict[str, Dict]
    ):
        self.providers = providers
        self.configs = configs
        self._clients: Dict[str, BaseLLMClient] = {}
        self._current_provider: Optional[str] = None
        
        # Initialize primary client
        self._initialize_clients()
    
    @property
    def provider_name(self) -> str:
        return f"fallback({self._current_provider or 'none'})"
    
    def _initialize_clients(self):
        """Initialize all available clients"""
        for provider in self.providers:
            try:
                config = self.configs.get(provider, {})
                client = LLMClientFactory.create(provider, **config)
                if client.is_available():
                    self._clients[provider] = client
                    if self._current_provider is None:
                        self._current_provider = provider
            except Exception as e:
                logger.warning(f"Failed to initialize {provider}: {e}")
    
    def _initialize_client(self):
        pass  # Clients initialized in __init__
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Generate with automatic fallback"""
        last_error = None
        
        for provider, client in self._clients.items():
            try:
                self._current_provider = provider
                return client.generate(messages, **kwargs)
            except Exception as e:
                logger.warning(f"{provider} failed: {e}")
                last_error = e
                continue
        
        raise RuntimeError(f"All providers failed. Last error: {last_error}")
    
    def get_langchain_model(self) -> BaseChatModel:
        """Get LangChain model from current provider"""
        if self._current_provider and self._current_provider in self._clients:
            return self._clients[self._current_provider].get_langchain_model()
        raise RuntimeError("No available LangChain model")
