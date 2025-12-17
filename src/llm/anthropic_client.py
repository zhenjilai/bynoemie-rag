"""
Anthropic Claude LLM Client

Supports Claude 3 Haiku, Sonnet, and Opus models.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any

from langchain_core.language_models import BaseChatModel

from .base import BaseLLMClient, LLMResponse, LLMClientFactory

logger = logging.getLogger(__name__)


@LLMClientFactory.register("anthropic")
class AnthropicClient(BaseLLMClient):
    """Anthropic Claude LLM client"""
    
    DEFAULT_MODEL = "claude-3-haiku-20240307"
    
    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: int = 30,
        **kwargs
    ):
        super().__init__(
            model=model or self.DEFAULT_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs
        )
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    
    @property
    def provider_name(self) -> str:
        return "anthropic"
    
    def _initialize_client(self):
        """Initialize Anthropic client"""
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key)
        
        return self._client
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Generate response using Anthropic Claude"""
        client = self._initialize_client()
        
        # Extract system message
        system_content = ""
        chat_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                chat_messages.append(msg)
        
        start_time = time.time()
        
        response = client.messages.create(
            model=self.model,
            max_tokens=kwargs.get('max_tokens', self.max_tokens),
            system=system_content,
            messages=chat_messages,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            provider=self.provider_name,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            },
            latency_ms=latency_ms,
            raw_response=response
        )
    
    def get_langchain_model(self) -> BaseChatModel:
        """Get LangChain ChatAnthropic model"""
        if self._langchain_model is None:
            from langchain_anthropic import ChatAnthropic
            
            self._langchain_model = ChatAnthropic(
                model=self.model,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        
        return self._langchain_model
    
    def is_available(self) -> bool:
        """Check if Anthropic is available"""
        if not self.api_key:
            return False
        
        try:
            self._initialize_client()
            return True
        except Exception as e:
            logger.warning(f"Anthropic not available: {e}")
            return False
