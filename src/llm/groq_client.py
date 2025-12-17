"""
Groq LLM Client

FREE tier available with excellent performance.
https://console.groq.com
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any

from langchain_core.language_models import BaseChatModel

from .base import BaseLLMClient, LLMResponse, LLMClientFactory

logger = logging.getLogger(__name__)


@LLMClientFactory.register("groq")
class GroqClient(BaseLLMClient):
    """Groq LLM client - FREE tier with fast inference"""
    
    DEFAULT_MODEL = "llama-3.1-70b-versatile"
    
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
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
    
    @property
    def provider_name(self) -> str:
        return "groq"
    
    def _initialize_client(self):
        """Initialize Groq client"""
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Get a FREE key at https://console.groq.com"
            )
        
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        
        return self._client
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Generate response using Groq"""
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
        """Get LangChain ChatGroq model"""
        if self._langchain_model is None:
            from langchain_groq import ChatGroq
            
            self._langchain_model = ChatGroq(
                model=self.model,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        
        return self._langchain_model
    
    def is_available(self) -> bool:
        """Check if Groq is available"""
        if not self.api_key:
            return False
        
        try:
            self._initialize_client()
            return True
        except Exception as e:
            logger.warning(f"Groq not available: {e}")
            return False
