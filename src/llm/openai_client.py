"""
OpenAI LLM Client

Supports GPT-4, GPT-4o, GPT-3.5-turbo models.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any

from langchain_core.language_models import BaseChatModel

from .base import BaseLLMClient, LLMResponse, LLMClientFactory

logger = logging.getLogger(__name__)


@LLMClientFactory.register("openai")
class OpenAIClient(BaseLLMClient):
    """OpenAI LLM client"""
    
    DEFAULT_MODEL = "gpt-4o-mini"
    
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
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        
        return self._client
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI"""
        client = self._initialize_client()
        
        start_time = time.time()
        
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get('temperature', self.temperature),
            max_tokens=kwargs.get('max_tokens', self.max_tokens),
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            provider=self.provider_name,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            latency_ms=latency_ms,
            raw_response=response
        )
    
    def get_langchain_model(self) -> BaseChatModel:
        """Get LangChain ChatOpenAI model"""
        if self._langchain_model is None:
            from langchain_openai import ChatOpenAI
            
            self._langchain_model = ChatOpenAI(
                model=self.model,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        
        return self._langchain_model
    
    def is_available(self) -> bool:
        """Check if OpenAI is available"""
        if not self.api_key:
            return False
        
        try:
            self._initialize_client()
            return True
        except Exception as e:
            logger.warning(f"OpenAI not available: {e}")
            return False
