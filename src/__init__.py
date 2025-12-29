"""
ByNoemie RAG Chatbot - Source Package

A production-ready RAG chatbot with:
- Multi-provider LLM support (Groq, OpenAI, Anthropic, Ollama)
- LangGraph workflow orchestration
- LangSmith tracing and monitoring
- Free-form vibe tag generation
- Vector-based product search
- Comprehensive evaluation metrics

Usage:
    from src.llm import create_llm_client
    from src.vibe_generator import VibeGeneratorWorkflow
    from src.rag import ProductDatabase, DataProcessor
    from src.evaluation import RAGEvaluator
    
    # Create LLM client
    llm = create_llm_client()
    
    # Generate vibes
    generator = VibeGeneratorWorkflow()
    result = generator.generate(
        product_name="Coco Dress",
        description="Sequin mini dress..."
    )
    
    # Search products
    db = ProductDatabase()
    results = db.search("romantic dinner dress")
    
    # Evaluate
    evaluator = RAGEvaluator()
    metrics = evaluator.evaluate_rag_system(...)
"""

__version__ = "1.0.0"
__author__ = "ByNoemie"

from . import llm
from . import prompt_engineering
from . import utils
from . import handlers
from . import vibe_generator

# These have heavier dependencies, import on demand
# from . import rag
# from . import evaluation

__all__ = [
    "llm",
    "prompt_engineering", 
    "utils",
    "handlers",
    "vibe_generator",
    # "rag",  # import separately: from src.rag import ProductDatabase
    # "evaluation",  # import separately: from src.evaluation import RAGEvaluator
]
