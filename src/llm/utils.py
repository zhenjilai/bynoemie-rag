"""
LLM Utilities Module

Helper functions for working with LLMs.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)


def parse_json_response(response: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM response, handling common formatting issues.
    
    Handles:
    - Markdown code blocks (```json ... ```)
    - Extra whitespace
    - Trailing commas (non-standard but common)
    """
    # Remove markdown code blocks
    response = response.strip()
    
    if '```json' in response:
        response = response.split('```json')[1].split('```')[0]
    elif '```' in response:
        parts = response.split('```')
        if len(parts) >= 2:
            response = parts[1]
    
    response = response.strip()
    
    # Try to find JSON object or array
    if response.startswith('{') or response.startswith('['):
        json_str = response
    else:
        # Find first { or [
        start_obj = response.find('{')
        start_arr = response.find('[')
        
        if start_obj == -1 and start_arr == -1:
            raise ValueError("No JSON found in response")
        
        if start_obj == -1:
            start = start_arr
            end = response.rfind(']') + 1
        elif start_arr == -1:
            start = start_obj
            end = response.rfind('}') + 1
        else:
            start = min(start_obj, start_arr)
            if start == start_obj:
                end = response.rfind('}') + 1
            else:
                end = response.rfind(']') + 1
        
        json_str = response[start:end]
    
    # Remove trailing commas (not valid JSON but LLMs do this)
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    return json.loads(json_str)


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Estimate token count for text.
    
    Uses tiktoken for accurate counts with OpenAI models,
    falls back to word-based estimate for others.
    """
    try:
        import tiktoken
        
        # Map model names to encodings
        encoding_map = {
            "gpt-4": "cl100k_base",
            "gpt-4o": "cl100k_base",
            "gpt-4o-mini": "cl100k_base",
            "gpt-3.5-turbo": "cl100k_base",
            "claude": "cl100k_base",  # Approximate
        }
        
        encoding_name = "cl100k_base"  # Default
        for key, enc in encoding_map.items():
            if key in model.lower():
                encoding_name = enc
                break
        
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
        
    except ImportError:
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    model: str = "gpt-4"
) -> str:
    """Truncate text to fit within token limit"""
    try:
        import tiktoken
        
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
        
    except ImportError:
        # Fallback: character-based truncation
        char_limit = max_tokens * 4
        return text[:char_limit] if len(text) > char_limit else text


def format_messages(
    system_prompt: str,
    user_prompt: str,
    history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """Format messages for chat completion"""
    messages = [{"role": "system", "content": system_prompt}]
    
    if history:
        messages.extend(history)
    
    messages.append({"role": "user", "content": user_prompt})
    
    return messages


def extract_code_blocks(text: str, language: str = None) -> List[str]:
    """Extract code blocks from markdown text"""
    if language:
        pattern = rf'```{language}\n(.*?)```'
    else:
        pattern = r'```(?:\w+)?\n(.*?)```'
    
    matches = re.findall(pattern, text, re.DOTALL)
    return [m.strip() for m in matches]


def clean_llm_output(text: str) -> str:
    """Clean common LLM output artifacts"""
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove common prefixes LLMs add
    prefixes_to_remove = [
        "Sure, here's",
        "Here's the",
        "I'll help you",
        "Of course!",
        "Certainly!",
    ]
    
    for prefix in prefixes_to_remove:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            if text.startswith(':'):
                text = text[1:].strip()
    
    return text


def validate_vibe_tags(
    tags: List[str],
    min_count: int = 3,
    max_count: int = 15
) -> List[str]:
    """Validate and clean vibe tags"""
    if not tags:
        return []
    
    # Clean tags
    cleaned = []
    for tag in tags:
        tag = tag.strip().lower()
        
        # Skip empty or too short
        if len(tag) < 2:
            continue
        
        # Skip tags that are too long
        if len(tag) > 50:
            continue
        
        # Skip duplicates
        if tag not in cleaned:
            cleaned.append(tag)
    
    # Enforce limits
    if len(cleaned) < min_count:
        logger.warning(f"Too few vibe tags: {len(cleaned)}, expected >= {min_count}")
    
    return cleaned[:max_count]
