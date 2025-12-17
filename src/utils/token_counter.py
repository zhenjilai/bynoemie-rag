"""
Token Counter Utility

Accurate token counting for various LLM providers.
"""

import logging
from typing import Dict, List, Union, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)


class TokenCounter:
    """
    Count tokens for different LLM providers.
    
    Supports:
    - OpenAI models (via tiktoken)
    - Anthropic Claude (approximate)
    - Llama models (approximate)
    """
    
    # Model to encoding mapping
    ENCODING_MAP = {
        "gpt-4": "cl100k_base",
        "gpt-4o": "cl100k_base",
        "gpt-4o-mini": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "claude": "cl100k_base",  # Approximate
        "llama": "cl100k_base",   # Approximate
    }
    
    def __init__(self):
        self._encodings: Dict = {}
    
    @lru_cache(maxsize=10)
    def _get_encoding(self, encoding_name: str):
        """Get or create tiktoken encoding"""
        try:
            import tiktoken
            return tiktoken.get_encoding(encoding_name)
        except ImportError:
            logger.warning("tiktoken not installed, using approximation")
            return None
    
    def _get_encoding_for_model(self, model: str) -> str:
        """Get encoding name for model"""
        model_lower = model.lower()
        
        for key, encoding in self.ENCODING_MAP.items():
            if key in model_lower:
                return encoding
        
        return "cl100k_base"  # Default
    
    def count(self, text: str, model: str = "gpt-4") -> int:
        """
        Count tokens in text for a specific model.
        
        Args:
            text: Text to count tokens for
            model: Model name (e.g., "gpt-4", "claude-3-haiku")
            
        Returns:
            Token count
        """
        encoding_name = self._get_encoding_for_model(model)
        encoding = self._get_encoding(encoding_name)
        
        if encoding is None:
            # Fallback: ~4 characters per token
            return len(text) // 4
        
        return len(encoding.encode(text))
    
    def count_messages(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4"
    ) -> int:
        """
        Count tokens in a list of chat messages.
        
        Accounts for message overhead (role tokens, etc.)
        """
        total = 0
        
        for message in messages:
            # Content tokens
            content = message.get("content", "")
            total += self.count(content, model)
            
            # Overhead per message (~4 tokens)
            total += 4
        
        # Base overhead (~3 tokens)
        total += 3
        
        return total
    
    def truncate_to_limit(
        self,
        text: str,
        max_tokens: int,
        model: str = "gpt-4"
    ) -> str:
        """
        Truncate text to fit within token limit.
        
        Returns:
            Truncated text
        """
        encoding_name = self._get_encoding_for_model(model)
        encoding = self._get_encoding(encoding_name)
        
        if encoding is None:
            # Fallback: character-based
            char_limit = max_tokens * 4
            return text[:char_limit] if len(text) > char_limit else text
        
        tokens = encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        return encoding.decode(tokens[:max_tokens])
    
    def split_into_chunks(
        self,
        text: str,
        chunk_size: int,
        overlap: int = 0,
        model: str = "gpt-4"
    ) -> List[str]:
        """
        Split text into chunks of specified token size.
        
        Args:
            text: Text to split
            chunk_size: Maximum tokens per chunk
            overlap: Token overlap between chunks
            model: Model for tokenization
            
        Returns:
            List of text chunks
        """
        encoding_name = self._get_encoding_for_model(model)
        encoding = self._get_encoding(encoding_name)
        
        if encoding is None:
            # Fallback: character-based chunking
            char_chunk = chunk_size * 4
            char_overlap = overlap * 4
            chunks = []
            start = 0
            while start < len(text):
                end = start + char_chunk
                chunks.append(text[start:end])
                start = end - char_overlap
            return chunks
        
        tokens = encoding.encode(text)
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunks.append(encoding.decode(chunk_tokens))
            start = end - overlap
        
        return chunks
    
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str
    ) -> float:
        """
        Estimate cost in USD for token usage.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name
            
        Returns:
            Estimated cost in USD
        """
        # Prices per 1M tokens (as of 2024)
        PRICING = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
            "claude-3-opus": {"input": 15.00, "output": 75.00},
            "claude-3-sonnet": {"input": 3.00, "output": 15.00},
            "claude-3-haiku": {"input": 0.25, "output": 1.25},
            "llama": {"input": 0.00, "output": 0.00},  # Free (local/Groq)
        }
        
        model_lower = model.lower()
        prices = None
        
        for key, pricing in PRICING.items():
            if key in model_lower:
                prices = pricing
                break
        
        if prices is None:
            return 0.0
        
        input_cost = (input_tokens / 1_000_000) * prices["input"]
        output_cost = (output_tokens / 1_000_000) * prices["output"]
        
        return input_cost + output_cost


# Singleton
_token_counter: Optional[TokenCounter] = None


def get_token_counter() -> TokenCounter:
    """Get singleton token counter"""
    global _token_counter
    if _token_counter is None:
        _token_counter = TokenCounter()
    return _token_counter


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Convenience function to count tokens"""
    return get_token_counter().count(text, model)
