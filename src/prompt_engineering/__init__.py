"""
Prompt Engineering Module for ByNoemie RAG Chatbot

Provides tools for:
- Template management with YAML support
- Few-shot example selection
- Prompt chaining with LangChain LCEL

Usage:
    from src.prompt_engineering import (
        get_template_manager,
        get_vibe_prompt_builder,
        PromptChainer,
        VibeGenerationChain
    )
    
    # Get templates
    tm = get_template_manager()
    system_prompt = tm.get_template("vibe_generator", "freeform")
    
    # Build vibe prompts
    builder = get_vibe_prompt_builder()
    prompt = builder.build_freeform_prompt(
        product_name="Coco Dress",
        description="Sequin mini dress..."
    )
    
    # Chain prompts
    chainer = PromptChainer(llm_client)
    chainer.add_step("analyze", prompt1)
    chainer.add_step("generate", prompt2)
    result = chainer.execute(text="input...")
"""

from .templates import (
    PromptTemplateManager,
    VibePromptBuilder,
    get_template_manager,
    get_vibe_prompt_builder
)

from .few_shot import (
    FewShotExample,
    FewShotManager,
    VibeExampleSelector,
    get_few_shot_manager
)

from .chainer import (
    ChainStep,
    PromptChainer,
    VibeGenerationChain
)


__all__ = [
    # Templates
    "PromptTemplateManager",
    "VibePromptBuilder",
    "get_template_manager",
    "get_vibe_prompt_builder",
    
    # Few-shot
    "FewShotExample",
    "FewShotManager",
    "VibeExampleSelector",
    "get_few_shot_manager",
    
    # Chaining
    "ChainStep",
    "PromptChainer",
    "VibeGenerationChain",
]
