"""
Vibe Generator Module for ByNoemie RAG Chatbot

Provides intelligent vibe tag generation using:
- LangGraph workflow orchestration
- LangSmith tracing and monitoring
- LangChain LLM integration
- Rule-based fallback and enhancement

Usage:
    from src.vibe_generator import create_vibe_generator, extract_vibes_from_product
    
    # Using LangGraph workflow (recommended)
    generator = create_vibe_generator()
    result = generator.generate(
        product_name="Coco Dress",
        description="Sequin mini dress...",
        colors="Black, Gold"
    )
    print(result["vibe_tags"])
    
    # Using rule-based extraction (fast, no API)
    vibes = extract_vibes_from_product({
        "product_name": "Coco Dress",
        "product_description": "Sequin mini dress..."
    })
"""

from .workflow import (
    VibeGeneratorState,
    VibeGeneratorWorkflow,
    build_vibe_generator_graph,
    create_vibe_generator
)

from .rules import (
    extract_vibes_from_product,
    extract_vibes_from_text,
    extract_vibes_from_material,
    extract_vibes_from_colors,
    get_vibe_scores,
    get_all_vibes,
    get_vibes_by_category,
    find_related_vibes,
    process_products_batch,
    VIBE_KEYWORDS,
    MATERIAL_VIBES,
    COLOR_VIBES
)


__all__ = [
    # Workflow
    "VibeGeneratorState",
    "VibeGeneratorWorkflow",
    "build_vibe_generator_graph",
    "create_vibe_generator",
    
    # Rule-based
    "extract_vibes_from_product",
    "extract_vibes_from_text",
    "extract_vibes_from_material",
    "extract_vibes_from_colors",
    "get_vibe_scores",
    "get_all_vibes",
    "get_vibes_by_category",
    "find_related_vibes",
    "process_products_batch",
    
    # Constants
    "VIBE_KEYWORDS",
    "MATERIAL_VIBES",
    "COLOR_VIBES",
]
