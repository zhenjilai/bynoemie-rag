"""
Ollama Local LLM Client

FREE local inference - no API keys needed.
Install: curl -fsSL https://ollama.com/install.sh | sh
Pull model: ollama pull llama3.2:3b
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any

import requests
from langchain_core.language_models import BaseChatModel

from .base import BaseLLMClient, LLMResponse, LLMClientFactory

logger = logging.getLogger(__name__)


@LLMClientFactory.register("ollama")
class OllamaClient(BaseLLMClient):
    """Ollama local LLM client - FREE, runs locally"""
    
    DEFAULT_MODEL = "llama3.2:3b"
    DEFAULT_BASE_URL = "http://localhost:11434"
    
    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: int = 60,
        **kwargs
    ):
        super().__init__(
            model=model or self.DEFAULT_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs
        )
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", self.DEFAULT_BASE_URL)
    
    @property
    def provider_name(self) -> str:
        return "ollama"
    
    def _initialize_client(self):
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            return True
        except Exception as e:
            raise ConnectionError(
                f"Ollama not available at {self.base_url}. "
                "Install: curl -fsSL https://ollama.com/install.sh | sh"
            )
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Generate response using Ollama"""
        self._initialize_client()
        
        start_time = time.time()
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get('temperature', self.temperature),
                    "num_predict": kwargs.get('max_tokens', self.max_tokens),
                }
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        
        data = response.json()
        latency_ms = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=data["message"]["content"],
            model=self.model,
            provider=self.provider_name,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
            },
            latency_ms=latency_ms,
            raw_response=data
        )
    
    def get_langchain_model(self) -> BaseChatModel:
        """Get LangChain ChatOllama model"""
        if self._langchain_model is None:
            from langchain_ollama import ChatOllama
            
            self._langchain_model = ChatOllama(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
            )
        
        return self._langchain_model
    
    def is_available(self) -> bool:
        """Check if Ollama is running locally"""
        try:
            self._initialize_client()
            return True
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False
    
    def list_models(self) -> List[str]:
        """List available models in Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []
    
    def pull_model(self, model_name: str = None) -> bool:
        """Pull a model from Ollama registry"""
        model = model_name or self.model
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model},
                timeout=300  # Model pulls can take time
            )
            response.raise_for_status()
            logger.info(f"Successfully pulled model: {model}")
            return True
        except Exception as e:
            logger.error(f"Failed to pull model {model}: {e}")
            return False
