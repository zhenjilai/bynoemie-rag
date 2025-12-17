"""
Prompt Chainer Module

Enables multi-step prompt chains for complex tasks.
Integrates with LangChain LCEL (LangChain Expression Language).
"""

import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
    RunnableSequence
)

logger = logging.getLogger(__name__)


@dataclass
class ChainStep:
    """A single step in a prompt chain"""
    name: str
    prompt: Union[ChatPromptTemplate, PromptTemplate, str]
    output_parser: Any = None
    transform_input: Optional[Callable] = None
    transform_output: Optional[Callable] = None
    retry_on_error: bool = True
    max_retries: int = 2


class PromptChainer:
    """
    Build and execute multi-step prompt chains.
    
    Example:
        chainer = PromptChainer(llm_client)
        
        chainer.add_step(
            name="extract",
            prompt="Extract key features from: {text}",
            output_parser=JsonOutputParser()
        )
        
        chainer.add_step(
            name="generate",
            prompt="Generate tags based on: {extract}",
            output_parser=JsonOutputParser()
        )
        
        result = chainer.execute(text="Product description...")
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.steps: List[ChainStep] = []
        self._chain = None
    
    def add_step(
        self,
        name: str,
        prompt: Union[ChatPromptTemplate, PromptTemplate, str],
        output_parser: Any = None,
        transform_input: Optional[Callable] = None,
        transform_output: Optional[Callable] = None,
        **kwargs
    ) -> "PromptChainer":
        """Add a step to the chain"""
        
        # Convert string to PromptTemplate
        if isinstance(prompt, str):
            prompt = PromptTemplate.from_template(prompt)
        
        step = ChainStep(
            name=name,
            prompt=prompt,
            output_parser=output_parser or StrOutputParser(),
            transform_input=transform_input,
            transform_output=transform_output,
            **kwargs
        )
        
        self.steps.append(step)
        self._chain = None  # Reset chain
        
        return self
    
    def build_chain(self) -> RunnableSequence:
        """Build the LangChain runnable chain"""
        if not self.steps:
            raise ValueError("No steps added to chain")
        
        if self.llm_client is None:
            raise ValueError("LLM client not set")
        
        llm = self.llm_client.get_langchain_model()
        
        # Build chain from steps
        chain = RunnablePassthrough()
        
        for i, step in enumerate(self.steps):
            # Input transformation
            if step.transform_input:
                chain = chain | RunnableLambda(step.transform_input)
            
            # Prompt -> LLM -> Parser
            step_chain = step.prompt | llm | step.output_parser
            
            # Output transformation
            if step.transform_output:
                step_chain = step_chain | RunnableLambda(step.transform_output)
            
            # Add step name to output
            chain = chain | RunnableLambda(
                lambda x, step_name=step.name, sc=step_chain: {
                    **x if isinstance(x, dict) else {"input": x},
                    step_name: sc.invoke(x)
                }
            )
        
        self._chain = chain
        return chain
    
    def execute(self, **inputs) -> Dict[str, Any]:
        """Execute the chain with given inputs"""
        if self._chain is None:
            self.build_chain()
        
        try:
            result = self._chain.invoke(inputs)
            return result
        except Exception as e:
            logger.error(f"Chain execution failed: {e}")
            raise
    
    async def aexecute(self, **inputs) -> Dict[str, Any]:
        """Execute the chain asynchronously"""
        if self._chain is None:
            self.build_chain()
        
        try:
            result = await self._chain.ainvoke(inputs)
            return result
        except Exception as e:
            logger.error(f"Async chain execution failed: {e}")
            raise


class VibeGenerationChain:
    """
    Pre-built chain for vibe generation workflow.
    
    Steps:
    1. Analyze product features
    2. Generate creative vibes
    3. Validate and format output
    """
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.chain = self._build_chain()
    
    def _build_chain(self) -> RunnableSequence:
        """Build the vibe generation chain"""
        llm = self.llm_client.get_langchain_model()
        
        # Step 1: Analyze product
        analyze_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a fashion analyst. Extract key attributes."),
            ("human", """Analyze this product:
Name: {product_name}
Type: {product_type}
Description: {description}
Colors: {colors}
Material: {material}

Extract:
1. Key visual features
2. Target occasions
3. Style category
4. Unique selling points

Return as JSON.""")
        ])
        
        # Step 2: Generate vibes
        generate_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a creative fashion copywriter. Generate evocative vibe tags.
Be creative and specific - think "midnight glamour" not just "evening"."""),
            ("human", """Based on this analysis:
{analysis}

Generate 8-12 creative vibe tags that capture:
- Occasion (when to wear)
- Mood (how it feels)
- Style aesthetic
- Unique features

Return JSON: {{"vibe_tags": [...], "mood_summary": "..."}}""")
        ])
        
        # Step 3: Validate
        validate_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a quality checker. Validate and improve vibe tags."),
            ("human", """Review these vibe tags for "{product_name}":
{vibes}

Ensure:
1. Tags are specific and evocative
2. No generic terms like "nice" or "good"
3. 8-12 unique tags
4. Relevant to the product

Return improved JSON: {{"vibe_tags": [...], "mood_summary": "..."}}""")
        ])
        
        # Build chain
        chain = (
            RunnablePassthrough()
            | RunnableParallel(
                analysis=analyze_prompt | llm | StrOutputParser(),
                product_name=lambda x: x["product_name"],
                product_type=lambda x: x.get("product_type", ""),
                description=lambda x: x.get("description", ""),
                colors=lambda x: x.get("colors", ""),
                material=lambda x: x.get("material", "")
            )
            | RunnableParallel(
                vibes=generate_prompt | llm | JsonOutputParser(),
                product_name=lambda x: x["product_name"],
                analysis=lambda x: x["analysis"]
            )
            | validate_prompt | llm | JsonOutputParser()
        )
        
        return chain
    
    def generate(
        self,
        product_name: str,
        product_type: str = "",
        description: str = "",
        colors: str = "",
        material: str = ""
    ) -> Dict[str, Any]:
        """Generate vibes for a product"""
        return self.chain.invoke({
            "product_name": product_name,
            "product_type": product_type,
            "description": description,
            "colors": colors,
            "material": material
        })
    
    async def agenerate(self, **kwargs) -> Dict[str, Any]:
        """Generate vibes asynchronously"""
        return await self.chain.ainvoke(kwargs)
