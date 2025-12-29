"""
ByNoemie RAG Chatbot - FastAPI Production API

Endpoints:
- POST /chat - Send message and get response
- POST /search - Search products
- GET /products - List all products
- GET /product/{id} - Get product details
- GET /health - Health check

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message", min_length=1)
    session_id: Optional[str] = Field(None, description="Session ID for conversation history")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Previous messages")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant response")
    intent: str = Field(..., description="Detected intent")
    products: List[Dict] = Field(default=[], description="Recommended products")
    session_id: str = Field(..., description="Session ID")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query", min_length=1)
    limit: int = Field(default=10, ge=1, le=50, description="Max results")
    category: Optional[str] = Field(None, description="Filter by category")
    occasion: Optional[str] = Field(None, description="Filter by occasion")

class ProductResponse(BaseModel):
    product_id: str
    product_name: str
    product_type: str
    price: float
    currency: str
    colors: str
    category: Optional[str] = None
    subcategory: Optional[str] = None
    vibe_tags: List[str] = []
    image_url: Optional[str] = None
    product_url: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    products_loaded: int

class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    rating: str = Field(..., description="'positive', 'negative', or 'neutral'")
    comment: Optional[str] = None

# =============================================================================
# DATA LOADING
# =============================================================================

# Global data stores
products_data: List[Dict] = []
stock_data: Dict = {}
policies_data: List[Dict] = []
sessions: Dict[str, List[Dict]] = {}

def load_data():
    """Load all data files"""
    global products_data, stock_data, policies_data
    
    # Load products
    product_paths = ["data/products/bynoemie_products.json", "data/products/output.json"]
    for path in product_paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                products_data = json.load(f)
                logger.info(f"Loaded {len(products_data)} products from {path}")
                break
    
    # Load stock
    stock_path = "data/stock/stock_inventory.json"
    if os.path.exists(stock_path):
        with open(stock_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                stock_data = {str(item['product_id']): item for item in data}
            else:
                stock_data = data
            logger.info(f"Loaded stock data for {len(stock_data)} products")
    
    # Load policies
    policy_path = "data/policies/policies.json"
    if os.path.exists(policy_path):
        with open(policy_path, 'r', encoding='utf-8') as f:
            policies_data = json.load(f)
            logger.info(f"Loaded {len(policies_data)} policies")

# =============================================================================
# CORE LOGIC (imported from app.py logic)
# =============================================================================

def get_openai_client():
    """Get OpenAI client"""
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return OpenAI(api_key=api_key)
    except Exception as e:
        logger.warning(f"OpenAI client not available: {e}")
    return None

def classify_intent(query: str, history: List[Dict] = None) -> Dict:
    """Classify user intent"""
    client = get_openai_client()
    q = query.lower().strip()
    
    # Keywords for basic classification
    if any(g in q for g in ['bye', 'goodbye', 'end']):
        return {"intent": "END_CONVERSATION"}
    if any(g in q for g in ['hello', 'hi', 'hey']):
        return {"intent": "GREETING"}
    if any(p in q for p in ['return', 'refund', 'shipping', 'policy', 'delivery']):
        return {"intent": "POLICY_QUESTION"}
    if any(s in q for s in ['in stock', 'available', 'size']):
        return {"intent": "STOCK_CHECK"}
    if any(cat in q for cat in ['shoes', 'heels', 'bags', 'dresses', 'dress', 'jumpsuit', 'tops']):
        return {"intent": "RECOMMENDATION"}
    if any(r in q for r in ['recommend', 'suggest', 'show me', 'looking for']):
        return {"intent": "RECOMMENDATION"}
    if any(p in q for p in ['how much', 'price', 'cost']):
        return {"intent": "PRICE_CHECK"}
    if any(t in q for t in ['thank', 'thanks']):
        return {"intent": "THANKS"}
    
    # Use LLM for better classification if available
    if client:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Classify intent: GREETING, RECOMMENDATION, STOCK_CHECK, PRODUCT_INFO, PRICE_CHECK, POLICY_QUESTION, THANKS, OFF_TOPIC, GENERAL. Return JSON: {\"intent\": \"...\"}"},
                    {"role": "user", "content": query}
                ],
                max_tokens=50,
                temperature=0.1
            )
            import re
            result = response.choices[0].message.content
            match = re.search(r'\{.*\}', result, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
    
    return {"intent": "GENERAL"}

def search_products_api(
    query: str, 
    limit: int = 10, 
    category: str = None,
    occasion: str = None
) -> List[Dict]:
    """Search products with filters"""
    q = query.lower()
    results = []
    
    # Category mapping
    CATEGORY_MAP = {
        "shoes": "Footwear", "heels": "Footwear", "footwear": "Footwear",
        "bags": "Accessories", "bag": "Accessories", "clutch": "Accessories",
        "dresses": "Clothing", "dress": "Clothing", "jumpsuit": "Clothing",
        "tops": "Clothing", "top": "Clothing", "sets": "Clothing"
    }
    
    # Detect category from query
    target_category = category
    if not target_category:
        for kw, cat in CATEGORY_MAP.items():
            if kw in q:
                target_category = cat
                break
    
    for p in products_data:
        score = 0
        
        # Category filter
        if target_category:
            p_cat = p.get('category', '')
            p_type = p.get('product_type', '').lower()
            
            if p_cat == target_category:
                score += 50
            elif target_category == "Footwear" and any(ft in p_type for ft in ['heel', 'sandal', 'pump']):
                score += 50
            elif target_category == "Accessories" and any(acc in p_type for acc in ['bag', 'clutch', 'tote']):
                score += 50
            elif target_category == "Clothing" and any(cl in p_type for cl in ['dress', 'jumpsuit', 'top', 'set']):
                score += 50
            else:
                continue  # Skip non-matching category
        
        # Text matching
        name = p.get('product_name', '').lower()
        vibes = ' '.join(p.get('vibe_tags', [])).lower()
        occasions = ' '.join(p.get('occasions', [])).lower()
        
        for word in q.split():
            if word in name: score += 20
            if word in vibes: score += 10
            if word in occasions: score += 10
        
        # Occasion filter
        if occasion and occasion.lower() in occasions:
            score += 30
        
        if score > 0:
            results.append((p, score))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in results[:limit]]

def generate_response(query: str, intent: str, products: List[Dict] = None) -> str:
    """Generate response based on intent"""
    client = get_openai_client()
    
    if intent == "GREETING":
        return "Hello! 👋 Welcome to ByNoemie. I'm here to help you find the perfect outfit. What are you looking for today?"
    
    if intent == "THANKS":
        return "You're welcome! 💕 Let me know if you need anything else."
    
    if intent == "END_CONVERSATION":
        return "Thank you for visiting ByNoemie! Have a wonderful day! 💫"
    
    if intent == "OFF_TOPIC":
        return "I'm your fashion assistant, so I specialize in style advice! 👗 Can I help you find something fabulous to wear?"
    
    if intent == "RECOMMENDATION" and products:
        names = [p.get('product_name', '') for p in products[:3]]
        if client:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Fashion stylist. Give SHORT recommendation (2 sentences). Mention specific product names."},
                        {"role": "user", "content": f"Query: {query}\nProducts: {', '.join(names)}"}
                    ],
                    max_tokens=150,
                    temperature=0.7
                )
                return response.choices[0].message.content
            except:
                pass
        return f"I found some great options for you! Check out the {names[0]} - it's absolutely stunning! ✨"
    
    if intent == "POLICY_QUESTION":
        # Search policies
        for policy in policies_data:
            if any(kw in query.lower() for kw in policy.get('keywords', [])):
                return policy.get('answer', "Please contact us for policy details.")
        return "For detailed policy information, please visit our website or contact customer service."
    
    return "I'd be happy to help! Could you tell me more about what you're looking for?"

# =============================================================================
# FASTAPI APP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting ByNoemie API...")
    load_data()
    yield
    # Shutdown
    logger.info("Shutting down ByNoemie API...")

app = FastAPI(
    title="ByNoemie Fashion Assistant API",
    description="AI-powered fashion chatbot and product search API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        products_loaded=len(products_data)
    )

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Send a message to the chatbot and get a response.
    
    - Supports conversation history via session_id
    - Returns detected intent and relevant products
    """
    import uuid
    
    # Get or create session
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []
    
    # Add user message to history
    sessions[session_id].append({"role": "user", "content": request.message})
    
    # Classify intent
    intent_result = classify_intent(request.message, sessions[session_id])
    intent = intent_result.get("intent", "GENERAL")
    
    # Search products if recommendation
    products = []
    if intent in ["RECOMMENDATION", "PRODUCT_INFO", "STOCK_CHECK"]:
        products = search_products_api(request.message, limit=10)
    
    # Generate response
    response_text = generate_response(request.message, intent, products)
    
    # Add assistant response to history
    sessions[session_id].append({"role": "assistant", "content": response_text})
    
    # Limit history size
    if len(sessions[session_id]) > 20:
        sessions[session_id] = sessions[session_id][-20:]
    
    # Format products for response
    formatted_products = []
    for p in products[:10]:
        formatted_products.append({
            "product_id": str(p.get("product_id", "")),
            "product_name": p.get("product_name", ""),
            "product_type": p.get("product_type", ""),
            "price": p.get("price_min", 0),
            "currency": p.get("price_currency", "MYR"),
            "colors": p.get("colors_available", ""),
            "category": p.get("category", ""),
            "subcategory": p.get("subcategory", ""),
            "vibe_tags": p.get("vibe_tags", []),
            "image_url": p.get("image_url_1", ""),
            "product_url": p.get("product_link", "")
        })
    
    return ChatResponse(
        response=response_text,
        intent=intent,
        products=formatted_products,
        session_id=session_id
    )

@app.post("/search", tags=["Products"])
async def search_products(request: SearchRequest):
    """
    Search products by query with optional filters.
    
    - Supports category filtering (Clothing, Footwear, Accessories)
    - Supports occasion filtering (party, wedding, casual, etc.)
    """
    products = search_products_api(
        query=request.query,
        limit=request.limit,
        category=request.category,
        occasion=request.occasion
    )
    
    return {
        "query": request.query,
        "count": len(products),
        "products": [
            {
                "product_id": str(p.get("product_id", "")),
                "product_name": p.get("product_name", ""),
                "product_type": p.get("product_type", ""),
                "price": p.get("price_min", 0),
                "currency": p.get("price_currency", "MYR"),
                "colors": p.get("colors_available", ""),
                "category": p.get("category", ""),
                "vibe_tags": p.get("vibe_tags", [])[:5],
                "image_url": p.get("image_url_1", "")
            }
            for p in products
        ]
    }

@app.get("/products", tags=["Products"])
async def list_products(
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(50, ge=1, le=100, description="Max results")
):
    """List all products with optional category filter"""
    results = products_data
    
    if category:
        results = [p for p in results if p.get("category", "").lower() == category.lower()]
    
    return {
        "count": len(results[:limit]),
        "products": [
            {
                "product_id": str(p.get("product_id", "")),
                "product_name": p.get("product_name", ""),
                "product_type": p.get("product_type", ""),
                "price": p.get("price_min", 0),
                "category": p.get("category", ""),
                "image_url": p.get("image_url_1", "")
            }
            for p in results[:limit]
        ]
    }

@app.get("/products/{product_id}", tags=["Products"])
async def get_product(product_id: str):
    """Get product details by ID"""
    for p in products_data:
        if str(p.get("product_id", "")) == product_id:
            # Get stock info
            stock_info = stock_data.get(product_id, {})
            
            return {
                "product_id": product_id,
                "product_name": p.get("product_name", ""),
                "product_type": p.get("product_type", ""),
                "description": p.get("product_description", ""),
                "price": p.get("price_min", 0),
                "currency": p.get("price_currency", "MYR"),
                "colors": p.get("colors_available", ""),
                "sizes": p.get("size_options", ""),
                "material": p.get("material", ""),
                "category": p.get("category", ""),
                "subcategory": p.get("subcategory", ""),
                "vibe_tags": p.get("vibe_tags", []),
                "occasions": p.get("occasions", []),
                "style_attributes": p.get("style_attributes", []),
                "mood_summary": p.get("mood_summary", ""),
                "ideal_for": p.get("ideal_for", ""),
                "image_url": p.get("image_url_1", ""),
                "product_url": p.get("product_link", ""),
                "in_stock": stock_info.get("in_stock", True),
                "variants": stock_info.get("variants", [])
            }
    
    raise HTTPException(status_code=404, detail="Product not found")

@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for a chat response"""
    feedback_path = "data/feedback/api_feedback.json"
    
    # Load existing feedback
    feedback_data = []
    if os.path.exists(feedback_path):
        with open(feedback_path, 'r') as f:
            feedback_data = json.load(f)
    
    # Add new feedback
    feedback_data.append({
        "session_id": request.session_id,
        "message_id": request.message_id,
        "rating": request.rating,
        "comment": request.comment,
        "timestamp": datetime.now().isoformat()
    })
    
    # Save
    os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
    with open(feedback_path, 'w') as f:
        json.dump(feedback_data, f, indent=2)
    
    return {"status": "success", "message": "Feedback recorded"}

@app.get("/categories", tags=["Products"])
async def get_categories():
    """Get all product categories with counts"""
    categories = {}
    for p in products_data:
        cat = p.get("category", "Uncategorized")
        categories[cat] = categories.get(cat, 0) + 1
    
    return {
        "categories": [
            {"name": name, "count": count}
            for name, count in sorted(categories.items())
        ]
    }

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENV", "production") == "development"
    )
