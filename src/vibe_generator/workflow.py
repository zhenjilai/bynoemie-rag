"""
Vibe Generator Workflow using LangGraph

This module implements a stateful, multi-step vibe generation workflow
using LangGraph for orchestration and LangSmith for tracing.

Workflow Steps:
1. Analyze Product - Extract key features
2. Generate Vibes - Create creative vibe tags  
3. Validate Output - Ensure quality and format
4. (Optional) Enhance - Add rule-based tags
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Annotated, TypedDict
from dataclasses import dataclass, field

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)


# =============================================================================
# STATE DEFINITION
# =============================================================================

class VibeGeneratorState(TypedDict):
    """State for vibe generation workflow"""
    # Input
    product_id: str
    product_name: str
    product_type: str
    description: str
    colors: str
    material: str
    price: float
    currency: str
    image_url: Optional[str]
    
    # Intermediate results
    analysis: Optional[Dict[str, Any]]
    raw_vibes: Optional[Dict[str, Any]]
    rule_based_vibes: Optional[List[str]]
    
    # Output - Existing vibe fields (KEEP - granular like Groq)
    vibe_tags: List[str]
    mood_summary: Optional[str]
    ideal_for: Optional[str]
    styling_tip: Optional[str]
    occasions: Optional[List[str]]
    
    # Output - NEW visual/structural fields
    category: Optional[str]
    subcategory: Optional[str]
    materials: Optional[List[str]]
    has_embellishment: Optional[bool]
    style_attributes: Optional[List[str]]
    silhouette: Optional[str]
    
    # Metadata
    errors: List[str]
    retry_count: int
    status: str  # pending, analyzing, generating, validating, complete, error


# =============================================================================
# NODE FUNCTIONS
# =============================================================================

def analyze_product(state: VibeGeneratorState) -> VibeGeneratorState:
    """
    Node 1: Analyze product to extract key features.
    
    Extracts:
    - Visual features
    - Target occasions
    - Style category
    - Unique selling points
    """
    logger.info(f"Analyzing product: {state['product_name']}")
    
    try:
        from src.llm import create_llm_client
        
        llm = create_llm_client()
        
        prompt = f"""Analyze this fashion product and extract key attributes:

Product: {state['product_name']}
Type: {state['product_type']}
Description: {state['description']}
Colors: {state['colors']}
Material: {state['material']}
Price: {state['price']} {state['currency']}

Extract as JSON:
{{
  "visual_features": ["feature1", "feature2", ...],
  "target_occasions": ["occasion1", "occasion2", ...],
  "style_category": "category",
  "unique_selling_points": ["usp1", "usp2", ...],
  "target_customer": "description of ideal customer"
}}"""

        response = llm.chat(
            system_prompt="You are a fashion analyst. Extract structured attributes from product descriptions. Return only valid JSON.",
            user_prompt=prompt
        )
        
        from src.llm import parse_json_response
        analysis = parse_json_response(response.content)
        
        return {
            **state,
            "analysis": analysis,
            "status": "analyzed"
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {
            **state,
            "errors": state.get("errors", []) + [f"Analysis error: {str(e)}"],
            "status": "error"
        }


def generate_vibes(state: VibeGeneratorState) -> VibeGeneratorState:
    """
    Node 2: Generate creative vibe tags AND visual attributes using LLM.
    
    Produces:
    - vibe_tags (granular: "wedding guest", "office chic", etc.)
    - mood_summary
    - ideal_for
    - styling_tip
    - occasions
    - category, subcategory, materials, has_embellishment, style_attributes, silhouette
    """
    logger.info(f"Generating vibes for: {state['product_name']}")
    
    try:
        from src.llm import create_llm_client
        
        llm = create_llm_client()
        
        # Build context from analysis
        analysis_context = ""
        if state.get("analysis"):
            analysis_context = f"""
Based on analysis:
- Visual Features: {', '.join(state['analysis'].get('visual_features', []))}
- Target Occasions: {', '.join(state['analysis'].get('target_occasions', []))}
- Style: {state['analysis'].get('style_category', 'N/A')}
- USPs: {', '.join(state['analysis'].get('unique_selling_points', []))}
"""
        
        # Enhanced system prompt for granular vibes + visual attributes
        system_prompt = """You are an expert fashion stylist and product analyst for a luxury boutique.
Your task is to generate DETAILED, GRANULAR product metadata.

For vibe_tags, use SPECIFIC, occasion-based descriptors like:
- Occasions: "wedding guest", "cocktail event", "romantic dinner", "garden party", "office chic", "night out", "gala", "brunch", "beach vacation", "date night"
- Styles: "effortlessly chic", "figure flattering", "day to night", "spring fresh", "autumn elegance", "summer vibes", "red carpet ready"
- Moods: "elegant", "romantic", "sophisticated", "glamorous", "minimalist", "edgy", "bold", "timeless", "bohemian", "feminine", "playful", "luxurious"

Be creative and specific. Avoid generic tags like just "nice" or "pretty".

Return ONLY valid JSON."""
        
        user_prompt = f"""Generate complete metadata for this fashion product:

Product: {state['product_name']}
Type: {state['product_type']}
Colors: {state['colors']}
Material: {state['material']}
Price: {state['price']} {state['currency']}
Description: {state['description']}
{analysis_context}

Return JSON with ALL these fields:

{{
    "vibe_tags": ["wedding guest", "elegant", "romantic", "figure flattering", "garden party", "effortlessly chic"],
    "mood_summary": "A stunning piece that embodies timeless elegance with a modern twist...",
    "ideal_for": "The sophisticated woman who wants to make a memorable impression at special occasions",
    "styling_tip": "Pair with strappy gold heels and delicate jewelry for a polished look",
    "occasions": ["wedding", "garden party", "cocktail event", "romantic dinner", "gala"],
    "category": "Clothing",
    "subcategory": "Dress",
    "materials": ["silk", "chiffon"],
    "has_embellishment": false,
    "style_attributes": ["sleeveless", "fitted", "midi length", "v-neck"],
    "silhouette": "A-line"
}}

IMPORTANT:
- vibe_tags: 6-10 GRANULAR, specific tags (not generic)
- mood_summary: 1-2 evocative sentences
- ideal_for: Who this is perfect for
- styling_tip: Practical styling advice
- occasions: 3-6 suitable occasions
- category: Clothing, Footwear, or Accessories
- subcategory: Dress, Heel, Bag, Top, Jumpsuit, Set, etc.
- materials: Specific materials (silk, sequin, leather, lace, velvet, chiffon, etc.)
- has_embellishment: true if sequins, beads, rhinestones, crystals, glitter
- style_attributes: Visual details (neckline, sleeves, fit, length, back style)
- silhouette: Shape (A-line, bodycon, empire, fit-and-flare, straight, mermaid)"""

        response = llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7
        )
        
        from src.llm import parse_json_response
        vibes = parse_json_response(response.content)
        
        return {
            **state,
            "raw_vibes": vibes,
            # Existing vibe fields
            "vibe_tags": vibes.get("vibe_tags", []),
            "mood_summary": vibes.get("mood_summary"),
            "ideal_for": vibes.get("ideal_for"),
            "styling_tip": vibes.get("styling_tip"),
            "occasions": vibes.get("occasions", []),
            # NEW visual/structural fields
            "category": vibes.get("category", "Clothing"),
            "subcategory": vibes.get("subcategory", ""),
            "materials": vibes.get("materials", []),
            "has_embellishment": vibes.get("has_embellishment", False),
            "style_attributes": vibes.get("style_attributes", []),
            "silhouette": vibes.get("silhouette", ""),
            "status": "generated"
        }
        
    except Exception as e:
        logger.error(f"Vibe generation failed: {e}")
        state["retry_count"] = state.get("retry_count", 0) + 1
        
        return {
            **state,
            "errors": state.get("errors", []) + [f"Generation error: {str(e)}"],
            "status": "error" if state["retry_count"] >= 3 else "retry"
        }


def apply_rule_based_vibes(state: VibeGeneratorState) -> VibeGeneratorState:
    """
    Node 3: Apply rule-based vibe extraction as enhancement.
    
    Adds tags based on keywords, materials, colors.
    """
    logger.info(f"Applying rule-based vibes for: {state['product_name']}")
    
    try:
        # Import rule-based generator
        from src.vibe_generator.rules import extract_vibes_from_product
        
        rule_vibes = extract_vibes_from_product({
            "product_name": state["product_name"],
            "product_type": state["product_type"],
            "product_description": state["description"],
            "colors_available": state["colors"],
            "material": state["material"]
        })
        
        # Merge with LLM vibes (LLM takes priority, rules add missing)
        existing = set(v.lower() for v in state.get("vibe_tags", []))
        enhanced_vibes = list(state.get("vibe_tags", []))
        
        for vibe in rule_vibes:
            if vibe.lower() not in existing:
                enhanced_vibes.append(vibe)
                existing.add(vibe.lower())
        
        return {
            **state,
            "rule_based_vibes": rule_vibes,
            "vibe_tags": enhanced_vibes[:12],  # Cap at 12
            "status": "enhanced"
        }
        
    except Exception as e:
        logger.warning(f"Rule-based enhancement failed: {e}")
        # Non-critical, continue with LLM vibes
        return {
            **state,
            "status": "enhanced"
        }


def validate_output(state: VibeGeneratorState) -> VibeGeneratorState:
    """
    Node 4: Validate and clean the final output.
    
    Ensures:
    - Minimum 5 tags
    - No duplicates
    - Proper formatting
    """
    logger.info(f"Validating output for: {state['product_name']}")
    
    vibe_tags = state.get("vibe_tags", [])
    
    # Clean and deduplicate
    seen = set()
    cleaned_tags = []
    
    for tag in vibe_tags:
        tag = tag.strip().lower()
        
        if len(tag) < 2 or len(tag) > 50:
            continue
        
        if tag in seen:
            continue
        
        seen.add(tag)
        cleaned_tags.append(tag)
    
    # Ensure minimum tags
    if len(cleaned_tags) < 5:
        # Add fallback tags based on product type
        fallbacks = {
            "dress": ["elegant", "feminine", "versatile", "chic", "stylish"],
            "heel": ["sophisticated", "elevated", "polished", "sleek", "refined"],
            "bag": ["essential", "versatile", "everyday", "practical", "chic"],
        }
        
        product_type = state.get("product_type", "").lower()
        default_tags = fallbacks.get(product_type, fallbacks["dress"])
        
        for tag in default_tags:
            if tag not in seen and len(cleaned_tags) < 5:
                cleaned_tags.append(tag)
                seen.add(tag)
    
    return {
        **state,
        "vibe_tags": cleaned_tags,
        "status": "complete"
    }


def handle_error(state: VibeGeneratorState) -> VibeGeneratorState:
    """
    Error handling node - provides fallback vibes and metadata.
    """
    logger.error(f"Error handling for: {state['product_name']}, errors: {state.get('errors', [])}")
    
    # Provide minimal fallback
    fallback_tags = ["elegant", "versatile", "stylish", "feminine", "chic"]
    
    # Detect category from product type
    product_type = state.get("product_type", "").lower()
    product_name = state.get("product_name", "").lower()
    
    category = "Clothing"
    subcategory = "Dress"
    
    if any(w in product_type or w in product_name for w in ['heel', 'sandal', 'shoe', 'pump']):
        category = "Footwear"
        subcategory = "Heel"
    elif any(w in product_type or w in product_name for w in ['bag', 'clutch', 'tote']):
        category = "Accessories"
        subcategory = "Bag"
    elif 'jumpsuit' in product_type or 'jumpsuit' in product_name:
        subcategory = "Jumpsuit"
    elif 'top' in product_type or 'blouse' in product_type:
        subcategory = "Top"
    elif 'set' in product_type:
        subcategory = "Set"
    
    return {
        **state,
        "vibe_tags": fallback_tags,
        "mood_summary": "A versatile piece for any occasion.",
        "ideal_for": None,
        "styling_tip": None,
        "occasions": [],
        "category": category,
        "subcategory": subcategory,
        "materials": [],
        "has_embellishment": False,
        "style_attributes": [],
        "silhouette": "",
        "status": "complete_with_fallback"
    }


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def should_retry(state: VibeGeneratorState) -> str:
    """Determine if we should retry generation"""
    if state.get("status") == "retry" and state.get("retry_count", 0) < 3:
        return "generate_vibes"
    elif state.get("status") == "error":
        return "handle_error"
    else:
        return "apply_rules"


def check_vibes_quality(state: VibeGeneratorState) -> str:
    """Check if generated vibes meet quality threshold"""
    vibes = state.get("vibe_tags", [])
    
    if len(vibes) >= 5:
        return "validate"
    else:
        return "apply_rules"


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def build_vibe_generator_graph() -> StateGraph:
    """
    Build the LangGraph workflow for vibe generation.
    
    Flow:
    START -> analyze -> generate -> route -> (apply_rules | validate | retry | error)
    """
    # Create graph
    workflow = StateGraph(VibeGeneratorState)
    
    # Add nodes
    workflow.add_node("analyze", analyze_product)
    workflow.add_node("generate_vibes", generate_vibes)
    workflow.add_node("apply_rules", apply_rule_based_vibes)
    workflow.add_node("validate", validate_output)
    workflow.add_node("handle_error", handle_error)
    
    # Set entry point
    workflow.set_entry_point("analyze")
    
    # Add edges
    workflow.add_edge("analyze", "generate_vibes")
    
    # Conditional routing after generation
    workflow.add_conditional_edges(
        "generate_vibes",
        should_retry,
        {
            "generate_vibes": "generate_vibes",  # Retry
            "apply_rules": "apply_rules",        # Success
            "handle_error": "handle_error"       # Give up
        }
    )
    
    workflow.add_edge("apply_rules", "validate")
    workflow.add_edge("validate", END)
    workflow.add_edge("handle_error", END)
    
    return workflow


# =============================================================================
# MAIN GENERATOR CLASS
# =============================================================================

class VibeGeneratorWorkflow:
    """
    Main class for vibe generation using LangGraph.
    
    Usage:
        generator = VibeGeneratorWorkflow()
        
        result = generator.generate(
            product_name="Coco Dress",
            description="Sequin mini dress...",
            colors="Black, Gold",
            material="Sequin"
        )
        
        print(result["vibe_tags"])
    """
    
    def __init__(self, enable_checkpointing: bool = False):
        self.graph = build_vibe_generator_graph()
        
        # Optional checkpointing for state recovery
        if enable_checkpointing:
            self.checkpointer = MemorySaver()
            self.app = self.graph.compile(checkpointer=self.checkpointer)
        else:
            self.app = self.graph.compile()
        
        # Setup LangSmith tracing if configured
        self._setup_tracing()
    
    def _setup_tracing(self):
        """Setup LangSmith tracing"""
        try:
            from config import settings
            
            if settings.langsmith.enabled and settings.langsmith.api_key:
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_API_KEY"] = settings.langsmith.api_key
                os.environ["LANGCHAIN_PROJECT"] = settings.langsmith.project_name
                
                logger.info(f"LangSmith tracing enabled for project: {settings.langsmith.project_name}")
        except Exception as e:
            logger.debug(f"LangSmith tracing not configured: {e}")
    
    def generate(
        self,
        product_id: str = "",
        product_name: str = "",
        product_type: str = "",
        description: str = "",
        colors: str = "",
        material: str = "",
        price: float = 0,
        currency: str = "MYR",
        image_url: str = ""
    ) -> Dict[str, Any]:
        """
        Generate vibe tags and visual attributes for a product.
        
        Returns:
            Dict with vibe_tags, mood_summary, category, materials, etc.
        """
        # Initialize state
        initial_state: VibeGeneratorState = {
            "product_id": product_id,
            "product_name": product_name,
            "product_type": product_type,
            "description": description,
            "colors": colors,
            "material": material,
            "price": price,
            "currency": currency,
            "image_url": image_url,
            "analysis": None,
            "raw_vibes": None,
            "rule_based_vibes": None,
            # Existing vibe fields
            "vibe_tags": [],
            "mood_summary": None,
            "ideal_for": None,
            "styling_tip": None,
            "occasions": None,
            # NEW visual/structural fields
            "category": None,
            "subcategory": None,
            "materials": None,
            "has_embellishment": None,
            "style_attributes": None,
            "silhouette": None,
            # Metadata
            "errors": [],
            "retry_count": 0,
            "status": "pending"
        }
        
        # Run workflow
        try:
            final_state = self.app.invoke(initial_state)
            
            return {
                "product_id": final_state["product_id"],
                "product_name": final_state["product_name"],
                # Existing vibe fields (granular)
                "vibe_tags": final_state["vibe_tags"],
                "mood_summary": final_state.get("mood_summary"),
                "ideal_for": final_state.get("ideal_for"),
                "styling_tip": final_state.get("styling_tip"),
                "occasions": final_state.get("occasions", []),
                # NEW visual/structural fields
                "category": final_state.get("category", "Clothing"),
                "subcategory": final_state.get("subcategory", ""),
                "materials": final_state.get("materials", []),
                "has_embellishment": final_state.get("has_embellishment", False),
                "style_attributes": final_state.get("style_attributes", []),
                "silhouette": final_state.get("silhouette", ""),
                # Metadata
                "status": final_state["status"],
                "errors": final_state.get("errors", [])
            }
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            return {
                "product_id": product_id,
                "product_name": product_name,
                "vibe_tags": ["elegant", "versatile", "stylish"],
                "mood_summary": None,
                "ideal_for": None,
                "styling_tip": None,
                "occasions": [],
                "category": "Clothing",
                "subcategory": "",
                "materials": [],
                "has_embellishment": False,
                "style_attributes": [],
                "silhouette": "",
                "status": "error",
                "errors": [str(e)]
            }
    
    async def agenerate(self, **kwargs) -> Dict[str, Any]:
        """Async version of generate"""
        initial_state = {
            "product_id": kwargs.get("product_id", ""),
            "product_name": kwargs.get("product_name", ""),
            "product_type": kwargs.get("product_type", ""),
            "description": kwargs.get("description", ""),
            "colors": kwargs.get("colors", ""),
            "material": kwargs.get("material", ""),
            "price": kwargs.get("price", 0),
            "currency": kwargs.get("currency", "MYR"),
            "image_url": kwargs.get("image_url", ""),
            "analysis": None,
            "raw_vibes": None,
            "rule_based_vibes": None,
            # Existing vibe fields
            "vibe_tags": [],
            "mood_summary": None,
            "ideal_for": None,
            "styling_tip": None,
            "occasions": None,
            # NEW visual/structural fields
            "category": None,
            "subcategory": None,
            "materials": None,
            "has_embellishment": None,
            "style_attributes": None,
            "silhouette": None,
            # Metadata
            "errors": [],
            "retry_count": 0,
            "status": "pending"
        }
        
        final_state = await self.app.ainvoke(initial_state)
        
        return {
            "product_id": final_state["product_id"],
            "product_name": final_state["product_name"],
            # Existing vibe fields
            "vibe_tags": final_state["vibe_tags"],
            "mood_summary": final_state.get("mood_summary"),
            "ideal_for": final_state.get("ideal_for"),
            "styling_tip": final_state.get("styling_tip"),
            "occasions": final_state.get("occasions", []),
            # NEW visual/structural fields
            "category": final_state.get("category", "Clothing"),
            "subcategory": final_state.get("subcategory", ""),
            "materials": final_state.get("materials", []),
            "has_embellishment": final_state.get("has_embellishment", False),
            "style_attributes": final_state.get("style_attributes", []),
            "silhouette": final_state.get("silhouette", ""),
            # Metadata
            "status": final_state["status"]
        }
    
    def generate_batch(
        self,
        products: List[Dict[str, Any]],
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """Generate vibes and metadata for multiple products"""
        results = []
        
        for i, product in enumerate(products):
            logger.info(f"Processing product {i+1}/{len(products)}: {product.get('product_name', 'Unknown')}")
            
            result = self.generate(
                product_id=str(product.get("product_id", "")),
                product_name=product.get("product_name", ""),
                product_type=product.get("product_type", ""),
                description=product.get("product_description", ""),
                colors=product.get("colors_available", ""),
                material=product.get("material", ""),
                price=float(product.get("price_min", 0)),
                currency=product.get("price_currency", "MYR"),
                image_url=product.get("image_url_1", "") or product.get("image_url", "")
            )
            
            results.append(result)
        
        return results


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_vibe_generator(
    enable_checkpointing: bool = False
) -> VibeGeneratorWorkflow:
    """Create a vibe generator workflow instance"""
    return VibeGeneratorWorkflow(enable_checkpointing=enable_checkpointing)