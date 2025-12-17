"""
Rule-Based Vibe Extraction

Keyword-based vibe extraction for fast, consistent tagging.
Used as fallback and enhancement for LLM-generated vibes.
"""

import re
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# VIBE TAXONOMY WITH KEYWORDS
# =============================================================================

VIBE_KEYWORDS = {
    # Occasions
    "romantic dinner": ["romantic", "dinner", "date", "evening", "intimate", "candlelit"],
    "night out": ["night", "party", "club", "sparkle", "sequin", "daring", "dance", "disco"],
    "wedding guest": ["wedding", "ceremony", "formal", "occasion", "celebration"],
    "beach vacation": ["beach", "vacation", "resort", "tropical", "summer", "holiday"],
    "office chic": ["office", "work", "professional", "business", "meeting", "corporate"],
    "brunch": ["brunch", "daytime", "casual", "weekend", "relaxed", "morning"],
    "garden party": ["garden", "outdoor", "floral", "spring", "tea", "afternoon"],
    "cocktail event": ["cocktail", "event", "soiree", "drinks", "reception"],
    "gala": ["gala", "black tie", "formal", "red carpet", "grand"],
    "date night": ["date", "romantic", "evening", "special", "intimate"],
    "girls night": ["girls", "friends", "fun", "party", "celebration"],
    "holiday party": ["holiday", "festive", "christmas", "new year", "celebration"],
    
    # Moods
    "elegant": ["elegant", "graceful", "refined", "sophisticated", "polished", "classy"],
    "romantic": ["romantic", "dreamy", "soft", "delicate", "feminine", "lovely"],
    "bold": ["bold", "daring", "statement", "striking", "dramatic", "fierce"],
    "minimalist": ["minimal", "simple", "clean", "understated", "sleek", "modern"],
    "glamorous": ["glamorous", "glam", "luxe", "sparkle", "glitter", "dazzling", "sequin"],
    "playful": ["playful", "fun", "flirty", "cute", "whimsical", "cheerful"],
    "sophisticated": ["sophisticated", "refined", "polished", "chic", "upscale"],
    "bohemian": ["boho", "bohemian", "free", "artistic", "eclectic", "hippie"],
    "edgy": ["edgy", "rock", "punk", "leather", "dark", "alternative"],
    "classic": ["classic", "timeless", "traditional", "iconic", "enduring"],
    "modern": ["modern", "contemporary", "current", "fresh", "new"],
    "feminine": ["feminine", "girly", "soft", "delicate", "pretty", "ladylike"],
    "sensual": ["sensual", "sexy", "alluring", "seductive", "sultry"],
    "confident": ["confident", "powerful", "strong", "bold", "empowering"],
    
    # Seasons
    "summer vibes": ["summer", "sunny", "bright", "warm", "tropical", "beach"],
    "spring fresh": ["spring", "fresh", "floral", "bloom", "pastel", "light"],
    "autumn elegance": ["autumn", "fall", "rich", "warm", "cozy", "earthy"],
    "festive": ["festive", "holiday", "christmas", "new year", "sparkle", "celebration"],
    "tropical": ["tropical", "exotic", "island", "paradise", "palm", "resort"],
    
    # Styles
    "timeless": ["timeless", "classic", "enduring", "eternal", "forever"],
    "effortlessly chic": ["effortless", "easy", "natural", "relaxed", "casual chic"],
    "figure flattering": ["flattering", "slimming", "curve", "bodycon", "fitted"],
    "day to night": ["versatile", "transitional", "day to night", "multi"],
    "statement piece": ["statement", "standout", "show-stopping", "head-turning"],
    "luxurious": ["luxurious", "luxury", "premium", "high-end", "opulent", "silk", "satin"],
    "vintage inspired": ["vintage", "retro", "classic", "old hollywood", "nostalgic"],
    "contemporary": ["contemporary", "modern", "current", "trendy", "now"],
}

# Material to vibe mapping
MATERIAL_VIBES = {
    "silk": ["luxurious", "elegant", "sensual", "sophisticated"],
    "satin": ["glamorous", "luxurious", "romantic", "elegant"],
    "lace": ["romantic", "feminine", "delicate", "elegant"],
    "sequin": ["glamorous", "night out", "festive", "bold"],
    "velvet": ["luxurious", "elegant", "autumn elegance", "sophisticated"],
    "leather": ["edgy", "bold", "confident", "modern"],
    "cotton": ["casual", "comfortable", "effortlessly chic", "everyday"],
    "linen": ["summer vibes", "relaxed", "beach vacation", "effortlessly chic"],
    "chiffon": ["romantic", "feminine", "elegant", "dreamy"],
    "jersey": ["comfortable", "versatile", "day to night", "effortlessly chic"],
    "mesh": ["edgy", "bold", "night out", "modern"],
    "tulle": ["romantic", "dreamy", "feminine", "wedding guest"],
}

# Color to vibe mapping
COLOR_VIBES = {
    "black": ["sophisticated", "elegant", "timeless", "versatile", "night out"],
    "white": ["elegant", "classic", "fresh", "minimalist", "bridal"],
    "red": ["bold", "confident", "romantic", "statement piece", "passionate"],
    "pink": ["romantic", "feminine", "playful", "soft", "girly"],
    "blue": ["classic", "sophisticated", "calm", "versatile"],
    "navy": ["sophisticated", "classic", "elegant", "professional"],
    "gold": ["glamorous", "luxurious", "festive", "statement piece"],
    "silver": ["glamorous", "modern", "festive", "night out"],
    "green": ["fresh", "natural", "elegant", "sophisticated"],
    "burgundy": ["elegant", "autumn elegance", "sophisticated", "romantic"],
    "nude": ["versatile", "elegant", "minimalist", "sophisticated"],
    "beige": ["versatile", "classic", "effortlessly chic", "neutral"],
    "coral": ["summer vibes", "playful", "fresh", "tropical"],
    "purple": ["bold", "luxurious", "creative", "sophisticated"],
}

# Product type defaults
PRODUCT_TYPE_VIBES = {
    "dress": ["feminine", "elegant", "versatile"],
    "gown": ["formal", "elegant", "glamorous", "gala"],
    "jumpsuit": ["modern", "bold", "versatile", "confident"],
    "romper": ["playful", "summer vibes", "casual", "fun"],
    "heel": ["sophisticated", "elegant", "polished"],
    "sandal": ["summer vibes", "casual", "beach vacation"],
    "bag": ["versatile", "essential", "polished"],
    "clutch": ["night out", "elegant", "glamorous"],
    "set": ["coordinated", "modern", "versatile"],
    "top": ["versatile", "everyday", "mix and match"],
    "skirt": ["feminine", "versatile", "ladylike"],
}


# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================

@dataclass
class VibeScore:
    """Vibe with confidence score"""
    vibe: str
    score: float
    source: str  # description, material, color, type


def extract_vibes_from_text(
    text: str,
    weight: float = 1.0
) -> Dict[str, float]:
    """
    Extract vibes from text using keyword matching.
    
    Args:
        text: Text to analyze
        weight: Weight multiplier for scores
        
    Returns:
        Dict of vibe -> score
    """
    if not text:
        return {}
    
    text_lower = text.lower()
    vibe_scores = {}
    
    for vibe, keywords in VIBE_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches > 0:
            # Score based on number of keyword matches
            score = min(1.0, matches * 0.3) * weight
            vibe_scores[vibe] = score
    
    return vibe_scores


def extract_vibes_from_material(material: str) -> Dict[str, float]:
    """Extract vibes based on material"""
    if not material:
        return {}
    
    material_lower = material.lower()
    vibe_scores = {}
    
    for mat, vibes in MATERIAL_VIBES.items():
        if mat in material_lower:
            for vibe in vibes:
                vibe_scores[vibe] = max(vibe_scores.get(vibe, 0), 0.8)
    
    return vibe_scores


def extract_vibes_from_colors(colors: str) -> Dict[str, float]:
    """Extract vibes based on colors"""
    if not colors:
        return {}
    
    colors_lower = colors.lower()
    vibe_scores = {}
    
    for color, vibes in COLOR_VIBES.items():
        if color in colors_lower:
            for vibe in vibes:
                vibe_scores[vibe] = max(vibe_scores.get(vibe, 0), 0.6)
    
    return vibe_scores


def extract_vibes_from_product_type(product_type: str) -> Dict[str, float]:
    """Extract default vibes based on product type"""
    if not product_type:
        return {}
    
    type_lower = product_type.lower()
    vibe_scores = {}
    
    for ptype, vibes in PRODUCT_TYPE_VIBES.items():
        if ptype in type_lower:
            for vibe in vibes:
                vibe_scores[vibe] = max(vibe_scores.get(vibe, 0), 0.4)
    
    return vibe_scores


def extract_vibes_from_product(product: Dict[str, Any]) -> List[str]:
    """
    Extract vibes from a product using all available signals.
    
    Args:
        product: Product dict with name, description, colors, material, type
        
    Returns:
        List of vibe tags (sorted by score)
    """
    all_scores: Dict[str, float] = {}
    
    # 1. Description (highest weight)
    description = product.get("product_description", "") or product.get("description", "")
    desc_vibes = extract_vibes_from_text(description, weight=1.5)
    for vibe, score in desc_vibes.items():
        all_scores[vibe] = max(all_scores.get(vibe, 0), score)
    
    # 2. Product name
    name = product.get("product_name", "") or product.get("name", "")
    name_vibes = extract_vibes_from_text(name, weight=1.2)
    for vibe, score in name_vibes.items():
        all_scores[vibe] = max(all_scores.get(vibe, 0), score)
    
    # 3. Material
    material = product.get("material", "")
    material_vibes = extract_vibes_from_material(material)
    for vibe, score in material_vibes.items():
        all_scores[vibe] = max(all_scores.get(vibe, 0), score)
    
    # 4. Colors
    colors = product.get("colors_available", "") or product.get("colors", "")
    color_vibes = extract_vibes_from_colors(colors)
    for vibe, score in color_vibes.items():
        all_scores[vibe] = max(all_scores.get(vibe, 0), score)
    
    # 5. Product type (lowest weight - defaults)
    product_type = product.get("product_type", "") or product.get("type", "")
    type_vibes = extract_vibes_from_product_type(product_type)
    for vibe, score in type_vibes.items():
        all_scores[vibe] = max(all_scores.get(vibe, 0), score)
    
    # Sort by score and return top vibes
    sorted_vibes = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return top 8 vibes with score > 0.3
    return [vibe for vibe, score in sorted_vibes[:8] if score > 0.3]


def get_vibe_scores(product: Dict[str, Any]) -> Dict[str, float]:
    """
    Get vibes with their scores (for weighted matching).
    
    Returns:
        Dict of vibe -> score (0-1)
    """
    all_scores: Dict[str, float] = {}
    
    # Same logic as extract_vibes_from_product but return scores
    description = product.get("product_description", "") or product.get("description", "")
    for vibe, score in extract_vibes_from_text(description, weight=1.5).items():
        all_scores[vibe] = max(all_scores.get(vibe, 0), score)
    
    name = product.get("product_name", "") or product.get("name", "")
    for vibe, score in extract_vibes_from_text(name, weight=1.2).items():
        all_scores[vibe] = max(all_scores.get(vibe, 0), score)
    
    material = product.get("material", "")
    for vibe, score in extract_vibes_from_material(material).items():
        all_scores[vibe] = max(all_scores.get(vibe, 0), score)
    
    colors = product.get("colors_available", "") or product.get("colors", "")
    for vibe, score in extract_vibes_from_colors(colors).items():
        all_scores[vibe] = max(all_scores.get(vibe, 0), score)
    
    product_type = product.get("product_type", "") or product.get("type", "")
    for vibe, score in extract_vibes_from_product_type(product_type).items():
        all_scores[vibe] = max(all_scores.get(vibe, 0), score)
    
    # Normalize scores
    if all_scores:
        max_score = max(all_scores.values())
        if max_score > 0:
            all_scores = {k: v / max_score for k, v in all_scores.items()}
    
    return all_scores


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_products_batch(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process multiple products and add vibe tags.
    
    Args:
        products: List of product dicts
        
    Returns:
        Products with added vibe_tags and vibe_scores
    """
    results = []
    
    for product in products:
        vibes = extract_vibes_from_product(product)
        scores = get_vibe_scores(product)
        
        result = {
            **product,
            "vibe_tags": vibes,
            "vibe_scores": {v: scores.get(v, 0) for v in vibes}
        }
        results.append(result)
    
    return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_all_vibes() -> List[str]:
    """Get list of all available vibes"""
    return list(VIBE_KEYWORDS.keys())


def get_vibes_by_category() -> Dict[str, List[str]]:
    """Get vibes organized by category"""
    return {
        "occasions": [
            "romantic dinner", "night out", "wedding guest", "beach vacation",
            "office chic", "brunch", "garden party", "cocktail event", "gala",
            "date night", "girls night", "holiday party"
        ],
        "moods": [
            "elegant", "romantic", "bold", "minimalist", "glamorous", "playful",
            "sophisticated", "bohemian", "edgy", "classic", "modern", "feminine",
            "sensual", "confident"
        ],
        "seasons": [
            "summer vibes", "spring fresh", "autumn elegance", "festive", "tropical"
        ],
        "styles": [
            "timeless", "effortlessly chic", "figure flattering", "day to night",
            "statement piece", "luxurious", "vintage inspired", "contemporary"
        ]
    }


def find_related_vibes(vibe: str) -> List[str]:
    """Find vibes related to the given vibe"""
    if vibe not in VIBE_KEYWORDS:
        return []
    
    keywords = set(VIBE_KEYWORDS[vibe])
    related = []
    
    for other_vibe, other_keywords in VIBE_KEYWORDS.items():
        if other_vibe == vibe:
            continue
        
        overlap = len(keywords.intersection(set(other_keywords)))
        if overlap >= 2:
            related.append(other_vibe)
    
    return related
