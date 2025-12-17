"""
Loads and manages all YAML configuration files with environment variable support.

API Keys are loaded from:
1. .env file (recommended for local development)
2. Environment variables (for production/Docker)
3. Falls back to empty string if not found

Usage:
    from config import settings
    
    # Access settings
    print(settings.llm.primary_provider)
    
    # Get API keys (automatically loaded from .env)
    import os
    groq_key = os.getenv("GROQ_API_KEY")
"""

import os
import yaml
import logging
import logging.config
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from functools import lru_cache

PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

def load_dotenv():
    """Load environment variables from .env file"""
    if not ENV_FILE.exists():
        # Check for .env.example and warn user
        example_file = PROJECT_ROOT / ".env.example"
        if example_file.exists():
            print(f"⚠️  No .env file found. Copy .env.example to .env and add your API keys:")
            print(f"   cp {example_file} {ENV_FILE}")
        return
    
    try:
        # Try python-dotenv first (recommended)
        from dotenv import load_dotenv as _load_dotenv
        _load_dotenv(ENV_FILE)
        print(f"✅ Loaded environment from {ENV_FILE}")
    except ImportError:
        # Fallback: manual parsing
        _load_env_manual(ENV_FILE)

def _load_env_manual(env_path: Path):
    """Manual .env parsing (fallback if python-dotenv not installed)"""
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            # Parse KEY=VALUE
            if '=' in line:
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip()
                # Remove quotes if present
                if value and value[0] in ('"', "'") and value[-1] == value[0]:
                    value = value[1:-1]
                # Only set if not already in environment
                if key and key not in os.environ:
                    os.environ[key] = value

# Load .env on module import
load_dotenv()

# Get config directory
CONFIG_DIR = Path(__file__).parent


def load_yaml(filename: str) -> Dict[str, Any]:
    """Load a YAML configuration file"""
    filepath = CONFIG_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Substitute environment variables
    return _substitute_env_vars(config)


def _substitute_env_vars(obj: Any) -> Any:
    """Recursively substitute ${VAR} with environment variables"""
    if isinstance(obj, str):
        # Handle ${VAR} or ${VAR:-default} syntax
        if obj.startswith('${') and '}' in obj:
            var_part = obj[2:obj.index('}')]
            if ':-' in var_part:
                var_name, default = var_part.split(':-', 1)
            else:
                var_name, default = var_part, ''
            return os.getenv(var_name, default)
        return obj
    elif isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    return obj


@dataclass
class LLMConfig:
    """LLM Provider Configuration"""
    primary_provider: str = "groq"
    fallback_chain: list = field(default_factory=lambda: ["groq", "openai", "ollama"])
    providers: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class EmbeddingsConfig:
    """Embeddings Configuration"""
    provider: str = "sentence_transformers"
    model: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    cache_enabled: bool = True


@dataclass
class VectorStoreConfig:
    """Vector Store Configuration"""
    provider: str = "chroma"
    collection_name: str = "bynoemie_products"
    persist_directory: str = "./data/embeddings/chroma_db"
    distance_metric: str = "cosine"


@dataclass
class VibeGeneratorConfig:
    """Vibe Generator Configuration"""
    mode: str = "freeform"
    tags_per_product: int = 10
    batch_size: int = 3
    max_retries: int = 3
    timeout: int = 60


@dataclass
class CacheConfig:
    """Cache Configuration"""
    enabled: bool = True
    ttl_seconds: int = 3600
    max_size: int = 1000
    backend: str = "disk"
    directory: str = "./data/cache"


@dataclass 
class LangSmithConfig:
    """LangSmith Tracing Configuration"""
    enabled: bool = False
    project_name: str = "bynoemie-rag-chatbot"
    trace_all: bool = True
    api_key: Optional[str] = None


class Settings:
    """Main settings class that loads all configurations"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Load configurations
        self._model_config = load_yaml('model_config.yaml')
        self._prompt_templates = load_yaml('prompt_templates.yaml')
        self._logging_config = load_yaml('logging_config.yaml')
        
        # Parse into dataclasses
        self.llm = self._parse_llm_config()
        self.embeddings = self._parse_embeddings_config()
        self.vector_store = self._parse_vector_store_config()
        self.vibe_generator = self._parse_vibe_generator_config()
        self.cache = self._parse_cache_config()
        self.langsmith = self._parse_langsmith_config()
        
        # Setup logging
        self._setup_logging()
        
        self._initialized = True
    
    def _parse_llm_config(self) -> LLMConfig:
        llm = self._model_config.get('llm', {})
        return LLMConfig(
            primary_provider=llm.get('primary_provider', 'groq'),
            fallback_chain=llm.get('fallback_chain', ['groq', 'openai', 'ollama']),
            providers=llm.get('providers', {})
        )
    
    def _parse_embeddings_config(self) -> EmbeddingsConfig:
        emb = self._model_config.get('embeddings', {})
        return EmbeddingsConfig(
            provider=emb.get('provider', 'sentence_transformers'),
            model=emb.get('model', 'all-MiniLM-L6-v2'),
            dimension=emb.get('dimension', 384),
            batch_size=emb.get('batch_size', 32),
            cache_enabled=emb.get('cache_enabled', True)
        )
    
    def _parse_vector_store_config(self) -> VectorStoreConfig:
        vs = self._model_config.get('vector_store', {})
        return VectorStoreConfig(
            provider=vs.get('provider', 'chroma'),
            collection_name=vs.get('collection_name', 'bynoemie_products'),
            persist_directory=vs.get('persist_directory', './data/embeddings/chroma_db'),
            distance_metric=vs.get('distance_metric', 'cosine')
        )
    
    def _parse_vibe_generator_config(self) -> VibeGeneratorConfig:
        vg = self._model_config.get('vibe_generator', {})
        workflow = vg.get('workflow', {})
        return VibeGeneratorConfig(
            mode=vg.get('mode', 'freeform'),
            tags_per_product=vg.get('tags_per_product', 10),
            batch_size=vg.get('batch_size', 3),
            max_retries=workflow.get('max_retries', 3),
            timeout=workflow.get('timeout', 60)
        )
    
    def _parse_cache_config(self) -> CacheConfig:
        cache = self._model_config.get('cache', {})
        return CacheConfig(
            enabled=cache.get('enabled', True),
            ttl_seconds=cache.get('ttl_seconds', 3600),
            max_size=cache.get('max_size', 1000),
            backend=cache.get('backend', 'disk'),
            directory=cache.get('directory', './data/cache')
        )
    
    def _parse_langsmith_config(self) -> LangSmithConfig:
        ls = self._model_config.get('langsmith', {})
        return LangSmithConfig(
            enabled=ls.get('enabled', False) or bool(os.getenv('LANGCHAIN_API_KEY')),
            project_name=ls.get('project_name', 'bynoemie-rag-chatbot'),
            trace_all=ls.get('trace_all', True),
            api_key=os.getenv('LANGCHAIN_API_KEY')
        )
    
    def _setup_logging(self):
        """Setup logging from configuration"""
        try:
            # Ensure log directories exist
            Path('./data/outputs').mkdir(parents=True, exist_ok=True)
            logging.config.dictConfig(self._logging_config)
        except Exception as e:
            # Fallback to basic logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            logging.warning(f"Failed to configure logging from YAML: {e}")
    
    def get_prompt(self, category: str, name: str) -> str:
        """Get a prompt template by category and name"""
        try:
            return self._prompt_templates[category][name]
        except KeyError:
            raise KeyError(f"Prompt not found: {category}.{name}")
    
    def get_few_shot_examples(self, name: str) -> list:
        """Get few-shot examples by name"""
        return self._prompt_templates.get('few_shot_examples', {}).get(name, [])
    
    def get_rate_limit(self, provider: str) -> Dict[str, int]:
        """Get rate limits for a provider"""
        return self._model_config.get('rate_limits', {}).get(provider, {
            'requests_per_minute': 60,
            'tokens_per_minute': 100000
        })


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Convenience exports
settings = get_settings()
