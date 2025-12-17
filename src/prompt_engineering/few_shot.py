"""
Few-Shot Examples Module

Manages few-shot examples for improved LLM performance.
Integrates with LangChain FewShotPromptTemplate.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import (
    SemanticSimilarityExampleSelector,
    MaxMarginalRelevanceExampleSelector
)

logger = logging.getLogger(__name__)


@dataclass
class FewShotExample:
    """A single few-shot example"""
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class FewShotManager:
    """Manages few-shot examples with selection strategies"""
    
    def __init__(self):
        self._examples: Dict[str, List[FewShotExample]] = {}
        self._load_examples()
    
    def _load_examples(self):
        """Load examples from config"""
        try:
            from config import settings
            raw_examples = settings.get_few_shot_examples("vibe_generation")
            
            vibe_examples = []
            for ex in raw_examples:
                vibe_examples.append(FewShotExample(
                    input_data=ex["input"],
                    output_data=ex["output"]
                ))
            
            self._examples["vibe_generation"] = vibe_examples
            logger.info(f"Loaded {len(vibe_examples)} vibe generation examples")
            
        except Exception as e:
            logger.warning(f"Could not load examples from config: {e}")
            self._examples = {}
    
    def get_examples(self, category: str) -> List[FewShotExample]:
        """Get all examples for a category"""
        return self._examples.get(category, [])
    
    def add_example(
        self,
        category: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ):
        """Add a new example"""
        if category not in self._examples:
            self._examples[category] = []
        
        self._examples[category].append(FewShotExample(
            input_data=input_data,
            output_data=output_data,
            metadata=metadata or {}
        ))
    
    def format_examples(
        self,
        category: str,
        input_template: str,
        output_template: str,
        max_examples: int = 3
    ) -> str:
        """Format examples as string for prompt injection"""
        examples = self.get_examples(category)[:max_examples]
        
        formatted = []
        for i, ex in enumerate(examples, 1):
            input_str = input_template.format(**ex.input_data)
            output_str = output_template.format(**ex.output_data)
            formatted.append(f"Example {i}:\nInput: {input_str}\nOutput: {output_str}")
        
        return "\n\n".join(formatted)
    
    def create_few_shot_template(
        self,
        category: str,
        example_template: PromptTemplate,
        prefix: str = "",
        suffix: str = "",
        max_examples: int = 3
    ) -> FewShotPromptTemplate:
        """Create a LangChain FewShotPromptTemplate"""
        examples = self.get_examples(category)[:max_examples]
        
        # Convert to LangChain format
        formatted_examples = []
        for ex in examples:
            formatted_examples.append({
                **ex.input_data,
                "output": str(ex.output_data)
            })
        
        return FewShotPromptTemplate(
            examples=formatted_examples,
            example_prompt=example_template,
            prefix=prefix,
            suffix=suffix,
            input_variables=example_template.input_variables
        )


class VibeExampleSelector:
    """Select relevant examples for vibe generation based on similarity"""
    
    def __init__(self, few_shot_manager: FewShotManager = None):
        self.manager = few_shot_manager or FewShotManager()
        self._embeddings = None
        self._selector = None
    
    def _initialize_selector(self):
        """Initialize semantic similarity selector"""
        if self._selector is not None:
            return
        
        try:
            from langchain_community.vectorstores import Chroma
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            self._embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            
            examples = self.manager.get_examples("vibe_generation")
            
            # Format examples for selector
            formatted = []
            for ex in examples:
                formatted.append({
                    "input": str(ex.input_data),
                    "output": str(ex.output_data)
                })
            
            if formatted:
                self._selector = SemanticSimilarityExampleSelector.from_examples(
                    formatted,
                    self._embeddings,
                    Chroma,
                    k=2
                )
                logger.info("Initialized semantic example selector")
            
        except Exception as e:
            logger.warning(f"Could not initialize semantic selector: {e}")
    
    def select_examples(
        self,
        query: str,
        k: int = 2
    ) -> List[FewShotExample]:
        """Select most relevant examples for a query"""
        self._initialize_selector()
        
        if self._selector is None:
            # Fallback: return first k examples
            return self.manager.get_examples("vibe_generation")[:k]
        
        try:
            selected = self._selector.select_examples({"input": query})
            
            # Convert back to FewShotExample format
            examples = []
            for sel in selected[:k]:
                examples.append(FewShotExample(
                    input_data={"text": sel["input"]},
                    output_data={"text": sel["output"]}
                ))
            
            return examples
            
        except Exception as e:
            logger.warning(f"Example selection failed: {e}")
            return self.manager.get_examples("vibe_generation")[:k]


# Singleton instances
_few_shot_manager: Optional[FewShotManager] = None


def get_few_shot_manager() -> FewShotManager:
    """Get singleton few-shot manager"""
    global _few_shot_manager
    if _few_shot_manager is None:
        _few_shot_manager = FewShotManager()
    return _few_shot_manager
