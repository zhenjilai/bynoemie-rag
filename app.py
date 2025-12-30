"""
ByNoemie RAG Chatbot - Full Featured Assistant

Features:
- LLM-based intent classification with CONTEXT AWARENESS
- Chat history (persists until user ends conversation)
- Order management (create, modify, cancel)
- Stock updates with ChromaDB sync
- Policy Q&A with RAG
- Streaming responses
- Per-response feedback saved to ChromaDB
- Dark elegant theme

Usage:
    streamlit run app.py
"""

import os
import json
import streamlit as st
import streamlit.components.v1 as components
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import random
import uuid

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="ByNoemie Fashion Assistant",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CUSTOM THEME - Dark Elegant
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=Montserrat:wght@300;400;500&display=swap');

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}
header {visibility: hidden;}

.stApp {
    background-color: #1a1a1a;
}

.main .block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

[data-testid="stChatMessageContent"] {
    background-color: #2a2a2a !important;
    border-radius: 12px;
    color: #f5f5f5;
}

.stChatInput > div {
    background-color: #2a2a2a !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 25px !important;
}

.stChatInput input {
    color: #f5f5f5 !important;
}

[data-testid="stSidebar"] {
    background-color: #1a1a1a;
    border-right: 1px solid #2a2a2a;
}

[data-testid="stSidebar"] * {
    color: #f5f5f5 !important;
}

.stButton > button {
    background-color: #2a2a2a;
    color: #d4a5a5;
    border: 1px solid #d4a5a5;
    border-radius: 20px;
    font-family: 'Montserrat', sans-serif;
    padding: 0.3rem 0.8rem;
    font-size: 0.85rem;
}

.stButton > button:hover {
    background-color: #d4a5a5;
    color: #1a1a1a;
}

h1, h2, h3 {
    font-family: 'Cormorant Garamond', serif !important;
    color: #f5f5f5 !important;
}

p, span, div, li {
    font-family: 'Montserrat', sans-serif;
}

[data-testid="stMetricValue"] {
    color: #d4a5a5 !important;
}

/* Feedback buttons - 10px gap */
.feedback-container {
    display: flex;
    gap: 10px;
    margin-top: 8px;
}

.feedback-container .stButton {
    margin: 0 !important;
}

.feedback-container .stButton > button {
    padding: 0.2rem 0.6rem;
    font-size: 1rem;
    min-width: 40px;
    height: 36px;
}

/* Fix column gaps for feedback buttons */
[data-testid="column"]:has(.feedback-btn) {
    padding-right: 10px !important;
    padding-left: 0 !important;
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 50px !important;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_products():
    paths = ["output.json", "data/products/output.json", "data/products/bynoemie_products.json"]
    for path in paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [p for p in data if not str(p.get('product_id', '')).startswith('prod_') and p.get('product_name')]
    return []


@st.cache_data(ttl=300, show_spinner=False)
def load_stock():
    path = "data/stock/stock_inventory.json"
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return {str(item['product_id']): item for item in data}
            return data
    return {}


@st.cache_data(ttl=3600, show_spinner=False)
def load_images():
    path = "data/products/bynoemie_products.json"
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {str(p['product_id']): {
                'image': p.get('image_url_1', ''),
                'link': p.get('product_link', '')
            } for p in data}
    return {}


@st.cache_data(ttl=3600, show_spinner=False)
def load_policies():
    path = "data/policies/policies.json"
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def get_api_key(key: str) -> str:
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except:
        pass
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except:
        pass
    return os.getenv(key, "")


# =============================================================================
# FEEDBACK SYSTEM - Save to ChromaDB
# =============================================================================
def get_feedback_collection():
    """Get or create feedback ChromaDB collection"""
    try:
        import chromadb
        from chromadb.config import Settings
        
        os.makedirs("data/embeddings/chroma_db", exist_ok=True)
        client = chromadb.PersistentClient(
            path="data/embeddings/chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        return client.get_or_create_collection(
            name="feedback",
            metadata={"description": "User feedback on responses"}
        )
    except:
        return None


def save_feedback(
    feedback_type: str,
    query: str,
    response: str,
    product_context: Optional[str] = None,
    message_id: str = None
):
    """Save feedback to ChromaDB and JSON"""
    feedback_id = message_id or "fb_{}_{}".format(
        datetime.now().strftime('%Y%m%d%H%M%S'),
        uuid.uuid4().hex[:6]
    )
    
    feedback_data = {
        "feedback_id": feedback_id,
        "type": feedback_type,
        "query": query,
        "response": response[:500],
        "product_context": product_context,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save to ChromaDB
    collection = get_feedback_collection()
    if collection:
        try:
            doc_text = "Query: {}\nResponse: {}\nFeedback: {}".format(
                query, response[:300], feedback_type
            )
            collection.upsert(
                ids=[feedback_id],
                documents=[doc_text],
                metadatas=[{
                    "type": feedback_type,
                    "query": query[:200],
                    "product_context": product_context or "",
                    "timestamp": feedback_data["timestamp"]
                }]
            )
        except Exception as e:
            print("ChromaDB feedback error: {}".format(e))
    
    # Also save to JSON
    try:
        os.makedirs("data/feedback", exist_ok=True)
        json_path = "data/feedback/feedback.json"
        existing = []
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        existing.append(feedback_data)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
    except:
        pass
    
    return feedback_id


# =============================================================================
# ORDER & USER DATABASE MANAGERS
# =============================================================================
def get_user_manager():
    """Get UserManager instance for user data operations"""
    if 'user_manager' not in st.session_state:
        try:
            from src.data_manager import UserManager
            st.session_state.user_manager = UserManager()
        except ImportError as e:
            print(f"UserManager import error: {e}")
            st.session_state.user_manager = None
    return st.session_state.user_manager


def get_order_manager():
    """Get OrderManager instance for order operations"""
    if 'order_manager' not in st.session_state:
        try:
            from src.data_manager import OrderManager, UserManager
            user_mgr = get_user_manager()
            st.session_state.order_manager = OrderManager(user_mgr)
        except ImportError as e:
            print(f"OrderManager import error: {e}")
            st.session_state.order_manager = None
    return st.session_state.order_manager


def get_database_manager():
    """Get combined DatabaseManager for easy access"""
    if 'db_manager' not in st.session_state:
        try:
            from src.data_manager import DatabaseManager
            st.session_state.db_manager = DatabaseManager()
        except ImportError as e:
            print(f"DatabaseManager import error: {e}")
            st.session_state.db_manager = None
    return st.session_state.db_manager


def get_policy_rag():
    if 'policy_rag' not in st.session_state:
        try:
            from src.policy_rag import PolicyRAG
            st.session_state.policy_rag = PolicyRAG()
        except ImportError as e:
            st.session_state.policy_rag = None
    return st.session_state.policy_rag


def reload_stock():
    load_stock.clear()
    return load_stock()


# =============================================================================
# LLM FUNCTIONS - ALL OPENAI (except vibe generation)
# =============================================================================
def get_openai_client():
    """Get OpenAI client for all LLM tasks"""
    api_key = get_api_key("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except:
        return None


def llm_classify_intent(
    query: str, 
    product_names: List[str],
    chat_history: List[Dict] = None,
    current_product: Optional[str] = None
) -> Dict:
    """
    Use OpenAI to classify intent WITH FULL CHAT HISTORY.
    Passes entire conversation context for accurate understanding.
    """
    client = get_openai_client()
    
    if not client:
        return fallback_classify(query, product_names, current_product)
    
    # Build system prompt
    system_prompt = """You are an intent classifier for ByNoemie fashion boutique chatbot.

You will receive the FULL CONVERSATION HISTORY. Use it to understand context.
If user says "this", "it", "that", "the item" - identify which product they're referring to from the conversation.

IMPORTANT: This is a FASHION BOUTIQUE chatbot. It should ONLY answer questions about:
- Products (dresses, jumpsuits, heels, bags, etc.)
- Stock availability
- Prices
- Store policies (shipping, refund, terms)
- Fashion recommendations
- Orders (create, modify, cancel, track)
- Customer profile

ANY question NOT related to fashion/products/store should be classified as OFF_TOPIC.

INTENTS:
- GREETING: Hello, hi, hey
- RECOMMENDATION: Show me, recommend, suggest, looking for, browse category
- STOCK_CHECK: In stock, available, sizes, do you have
- PRODUCT_INFO: Tell me about, what color, what material, describe
- PRICE_CHECK: How much, price, cost
- POLICY_QUESTION: Returns, refunds, shipping, terms
- ORDER_CREATE: I want to order, buy, purchase, place order
- ORDER_MODIFY: Change my order, modify order, update size/color/quantity
- ORDER_CANCEL: Cancel order, delete order, remove order
- ORDER_TRACK: Track order, where is my order, order status, check order
- ORDER_CONFIRM: User types exactly "ORDER", "DELETE", or "CHANGE" to confirm
- USER_PROFILE: My profile, my account, my info, customer info, update address
- END_CONVERSATION: Goodbye, bye
- THANKS: Thank you
- OFF_TOPIC: Not fashion/shopping related
- GENERAL: Other fashion questions

IMPORTANT RULES:
1. If user types EXACTLY "ORDER", "DELETE", or "CHANGE" (case-insensitive) -> ORDER_CONFIRM
2. Order IDs look like "ORD-001", "ORD-002", etc. Extract them if mentioned.
3. If user refers to "this dress", "it" -> find product from conversation context
4. For order actions, extract order_id if user mentions it

AVAILABLE PRODUCTS: {}

Return ONLY JSON:
{{"intent": "INTENT_NAME", "product_mentioned": "product name or null", "size": "size or null", "color": "color or null", "order_id": "ORD-XXX or null", "confirm_type": "ORDER/DELETE/CHANGE or null"}}""".format(', '.join(product_names[:20]))

    # Build messages with full chat history
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add chat history for context (last 10 messages max)
    if chat_history:
        recent_history = chat_history[-10:]
        for msg in recent_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                messages.append({"role": role, "content": content[:500]})
    
    # Add current query
    messages.append({"role": "user", "content": "Classify this query: {}".format(query)})
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=150,
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        import re
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed
        return {"intent": "GENERAL"}
    except Exception as e:
        print("OpenAI classification error: {}".format(e))
        return fallback_classify(query, product_names, current_product)


def fallback_classify(query: str, product_names: List[str], current_product: Optional[str] = None) -> Dict:
    """Fallback keyword-based classification with context"""
    q = query.lower().strip()
    
    # Check for references to current product
    ref_words = ['this', 'it', 'that', 'the item', 'same', 'above']
    refers_to_current = any(word in q for word in ref_words)
    
    # Find product mention
    product_mentioned = None
    for name in product_names:
        if name.lower() in q:
            product_mentioned = name
            break
    
    # If referring to current product
    if not product_mentioned and refers_to_current and current_product:
        product_mentioned = current_product
    
    # Extract size
    size = None
    size_map = {
        'xs': 'XS', 'extra small': 'XS',
        's size': 'S', 'size s': 'S', 'small': 'S',
        'm size': 'M', 'size m': 'M', 'medium': 'M',
        'l size': 'L', 'size l': 'L', 'large': 'L',
        'xl': 'XL', 'extra large': 'XL',
        'free size': 'Free Size'
    }
    for pattern, sz in size_map.items():
        if pattern in q:
            size = sz
            break
    
    # Extract color
    color = None
    colors = ['black', 'white', 'beige', 'red', 'blue', 'pink', 'green', 'brown', 'cream', 'navy', 'gold']
    for c in colors:
        if c in q:
            color = c.capitalize()
            break
    
    result = {"product_mentioned": product_mentioned, "size": size, "color": color}
    
    # Fashion/store related keywords
    fashion_keywords = [
        'dress', 'jumpsuit', 'top', 'skirt', 'pants', 'heel', 'bag', 'outfit', 'wear',
        'stock', 'available', 'size', 'price', 'cost', 'order', 'buy', 'purchase',
        'ship', 'delivery', 'refund', 'return', 'exchange', 'policy',
        'recommend', 'suggest', 'show', 'looking for', 'style', 'fashion',
        'wedding', 'party', 'dinner', 'date', 'occasion', 'formal', 'casual',
        'romantic', 'elegant', 'chic', 'color', 'colour', 'fabric', 'material',
        'shoes', 'shoe', 'footwear', 'sandal', 'purse', 'clutch', 'sets', 'set',
        'this dress', 'this one', 'that one', 'it'
    ]
    
    # Broad category keywords that should trigger RECOMMENDATION
    category_keywords = [
        'shoes', 'shoe', 'heels', 'heel', 'footwear', 'sandal', 'sandals',
        'bags', 'bag', 'purse', 'clutch', 'handbag',
        'dresses', 'dress', 'gown',
        'jumpsuits', 'jumpsuit', 'romper',
        'tops', 'top', 'blouse',
        'sets', 'set', 'co-ord'
    ]
    
    # Specific attribute questions - should be PRODUCT_INFO
    attribute_questions = [
        'what color', 'what colour', 'what colors', 'what colours',
        'what size', 'what sizes', 'size guide',
        'what material', 'what fabric', 'made of', 'made from',
        'how does it fit', 'fit like', 'true to size'
    ]
    
    # Check if query is fashion-related
    is_fashion_related = any(kw in q for kw in fashion_keywords) or product_mentioned
    
    # Check if query is asking for a category
    is_category_query = any(cat in q for cat in category_keywords)
    
    # Check if asking about specific product attributes
    is_attribute_question = any(attr in q for attr in attribute_questions)
    
    # Extract order ID if mentioned (e.g., ORD-001, ord-002)
    import re
    order_id_match = re.search(r'ord-?\d{3}', q, re.IGNORECASE)
    if order_id_match:
        order_id = order_id_match.group().upper()
        if 'ORD' in order_id and '-' not in order_id:
            order_id = order_id.replace('ORD', 'ORD-')
        result["order_id"] = order_id
    
    # Check for exact confirmation keywords (case-insensitive)
    q_stripped = query.strip().upper()
    
    # Determine intent - ORDER CONFIRMATIONS FIRST
    if q_stripped == "ORDER":
        result["intent"] = "ORDER_CONFIRM"
        result["confirm_type"] = "ORDER"
    elif q_stripped == "DELETE":
        result["intent"] = "ORDER_CONFIRM"
        result["confirm_type"] = "DELETE"
    elif q_stripped == "CHANGE":
        result["intent"] = "ORDER_CONFIRM"
        result["confirm_type"] = "CHANGE"
    elif any(g in q for g in ['bye', 'goodbye', 'end chat']):
        result["intent"] = "END_CONVERSATION"
    elif any(g in q for g in ['hello', 'hi', 'hey']) and len(q.split()) < 5:
        result["intent"] = "GREETING"
    elif any(p in q for p in ['return policy', 'refund', 'exchange policy', 'shipping policy', 'delivery time']):
        result["intent"] = "POLICY_QUESTION"
    # Order tracking - check before other order intents
    elif any(t in q for t in ['track order', 'track my order', 'where is my order', 'order status', 'check order', 'check my order']):
        result["intent"] = "ORDER_TRACK"
    # Order cancellation
    elif any(c in q for c in ['cancel order', 'cancel my order', 'delete order', 'remove order']):
        result["intent"] = "ORDER_CANCEL"
    # Order modification
    elif any(m in q for m in ['change order', 'change my order', 'modify order', 'update order', 'edit order']):
        result["intent"] = "ORDER_MODIFY"
    # Order creation - must have product context
    elif any(o in q for o in ['i want to order', 'like to order', 'would like to order', 'place order', 'buy this', 'purchase']) and (product_mentioned or refers_to_current):
        result["intent"] = "ORDER_CREATE"
    # User profile
    elif any(u in q for u in ['my profile', 'my account', 'my info', 'my address', 'update profile', 'customer info']):
        result["intent"] = "USER_PROFILE"
    elif is_attribute_question:
        result["intent"] = "PRODUCT_INFO"
    elif any(s in q for s in ['in stock', 'available', 'do you have']) and not is_category_query:
        result["intent"] = "STOCK_CHECK"
    elif any(p in q for p in ['how much', 'price', 'cost']) and is_fashion_related:
        result["intent"] = "PRICE_CHECK"
    elif any(i in q for i in ['tell me about', 'what is', 'describe']) and product_mentioned:
        result["intent"] = "PRODUCT_INFO"
    elif is_category_query or any(r in q for r in ['recommend', 'suggest', 'show me', 'looking for', 'show', 'what do you have', 'browse']):
        result["intent"] = "RECOMMENDATION"
    elif any(t in q for t in ['thank', 'thanks']):
        result["intent"] = "THANKS"
    elif not is_fashion_related:
        result["intent"] = "OFF_TOPIC"
    else:
        result["intent"] = "GENERAL"
    
    return result


def stream_response(client, messages, placeholder):
    """Stream response using OpenAI"""
    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300,
            temperature=0.7,
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                placeholder.markdown(full_response + "▌")
        
        placeholder.markdown(full_response)
        return full_response
    except Exception as e:
        print("OpenAI streaming error: {}".format(e))
        return None


def llm_generate_response(query: str, intent: str, context: str, use_stream: bool = True) -> str:
    """Generate response using OpenAI"""
    client = get_openai_client()
    if not client:
        return None
    
    prompts = {
        "GREETING": "You're a warm fashion assistant for ByNoemie. Brief welcoming greeting.",
        "RECOMMENDATION": "Fashion stylist. SHORT recommendations (2-3 sentences). Mention product names.",
        "STOCK_CHECK": "Provide stock information clearly and professionally.",
        "PRODUCT_INFO": "Describe the product elegantly in 2-3 sentences.",
        "PRICE_CHECK": "State the price clearly.",
        "POLICY_QUESTION": "Answer the policy question helpfully.",
        "ORDER_CREATE": "Confirm order details professionally.",
        "END_CONVERSATION": "Gracious farewell, thank them.",
        "THANKS": "Respond warmly.",
        "GENERAL": "Be helpful and professional."
    }
    
    messages = [
        {"role": "system", "content": prompts.get(intent, prompts["GENERAL"])},
        {"role": "user", "content": "Query: {}\n\nContext: {}".format(query, context)}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print("OpenAI response error: {}".format(e))
        return None


# =============================================================================
# SEARCH & DATA FUNCTIONS
# =============================================================================
def llm_understand_query(query: str, client) -> Dict:
    """
    Use LLM to understand what the user is looking for.
    Returns structured info about category, style, occasion, etc.
    """
    if not client:
        return {}
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Extract shopping intent from the query. Return JSON only:
{
    "category": "Clothing/Footwear/Accessories or null",
    "subcategory": "Dress/Heel/Bag/Top/Jumpsuit or null",
    "product_types": ["Maxi Dress", "Stiletto Heel", etc. or empty],
    "occasion": "party/wedding/dinner/casual/work or null",
    "style": "romantic/elegant/chic/bold or null",
    "color": "black/white/red or null",
    "material": "silk/leather/sequin or null",
    "keywords": ["additional", "search", "terms"]
}

Examples:
- "show me shoes" → {"category": "Footwear", "subcategory": "Heel", "product_types": ["Heel", "Sandal"]}
- "party dresses" → {"category": "Clothing", "subcategory": "Dress", "occasion": "party"}
- "silk dress" → {"category": "Clothing", "subcategory": "Dress", "material": "silk"}
- "bling bling heels" → {"category": "Footwear", "material": "sequin", "keywords": ["embellishment"]}"""
                },
                {"role": "user", "content": query}
            ],
            max_tokens=200,
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        import re
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print("Query understanding error: {}".format(e))
    return {}


def search_products(query: str, products: List[Dict], n: int = 8, query_understanding: Dict = None) -> List[Dict]:
    """
    Search products with enhanced metadata support.
    Uses category, subcategory, materials, vibes, occasions, etc.
    """
    if not query or len(query.strip()) < 2:
        return []
    
    q = query.lower()
    
    # === SYNONYM MAPPING ===
    SYNONYMS = {
        "shoes": ["heel", "heels", "sandal", "sandals", "footwear", "pumps"],
        "shoe": ["heel", "heels", "sandal", "sandals", "footwear", "pumps"],
        "footwear": ["heel", "heels", "sandal", "sandals"],
        "bags": ["bag", "clutch", "purse", "handbag", "tote"],
        "bag": ["clutch", "purse", "handbag", "tote"],
        "purse": ["bag", "clutch", "handbag"],
        "tops": ["top", "blouse", "shirt", "camisole"],
        "top": ["blouse", "shirt", "camisole"],
        "pants": ["trouser", "trousers", "slacks"],
        "bottoms": ["pants", "skirt", "shorts", "trouser"],
        "dresses": ["dress", "gown", "maxi", "mini"],
        "gown": ["dress", "evening dress", "formal dress"],
        "jumpsuits": ["jumpsuit", "romper", "playsuit"],
        "sparkly": ["sequin", "glitter", "rhinestone", "crystal", "bling"],
        "bling": ["sequin", "glitter", "rhinestone", "crystal", "embellishment"],
        "shiny": ["sequin", "satin", "silk", "metallic"],
    }
    
    # === CATEGORY MAPPING ===
    CATEGORY_MAP = {
        "shoes": {"category": "Footwear", "subcategories": ["Heel", "Sandal"]},
        "shoe": {"category": "Footwear", "subcategories": ["Heel", "Sandal"]},
        "heels": {"category": "Footwear", "subcategories": ["Heel"]},
        "heel": {"category": "Footwear", "subcategories": ["Heel"]},
        "sandals": {"category": "Footwear", "subcategories": ["Sandal"]},
        "footwear": {"category": "Footwear", "subcategories": ["Heel", "Sandal"]},
        "bags": {"category": "Accessories", "subcategories": ["Bag", "Clutch", "Handbag"]},
        "bag": {"category": "Accessories", "subcategories": ["Bag", "Clutch", "Handbag"]},
        "purse": {"category": "Accessories", "subcategories": ["Bag", "Clutch", "Handbag"]},
        "clutch": {"category": "Accessories", "subcategories": ["Clutch"]},
        "dresses": {"category": "Clothing", "subcategories": ["Dress"]},
        "dress": {"category": "Clothing", "subcategories": ["Dress"]},
        "jumpsuits": {"category": "Clothing", "subcategories": ["Jumpsuit"]},
        "jumpsuit": {"category": "Clothing", "subcategories": ["Jumpsuit"]},
        "tops": {"category": "Clothing", "subcategories": ["Top", "Blouse"]},
        "top": {"category": "Clothing", "subcategories": ["Top", "Blouse"]},
        "sets": {"category": "Clothing", "subcategories": ["Set", "Co-ord"]},
        "set": {"category": "Clothing", "subcategories": ["Set", "Co-ord"]},
    }
    
    # === MATERIAL KEYWORDS ===
    MATERIAL_KEYWORDS = {
        "silk": ["silk", "satin", "silky"],
        "leather": ["leather", "faux leather", "pu leather"],
        "sequin": ["sequin", "glitter", "sparkle", "bling", "rhinestone", "crystal", "beaded"],
        "lace": ["lace", "lacey"],
        "velvet": ["velvet", "velour"],
        "cotton": ["cotton", "linen"],
        "chiffon": ["chiffon", "sheer", "mesh"],
    }
    
    # Extract from LLM understanding
    target_category = None
    target_subcategories = []
    target_materials = []
    target_occasions = []
    target_vibes = []
    extra_keywords = []
    
    if query_understanding:
        target_category = query_understanding.get('category')
        if query_understanding.get('subcategory'):
            target_subcategories.append(query_understanding['subcategory'])
        if query_understanding.get('product_types'):
            target_subcategories.extend(query_understanding['product_types'])
        if query_understanding.get('material'):
            target_materials.append(query_understanding['material'])
        if query_understanding.get('occasion'):
            target_occasions.append(query_understanding['occasion'])
        if query_understanding.get('style'):
            target_vibes.append(query_understanding['style'])
        if query_understanding.get('keywords'):
            extra_keywords.extend(query_understanding['keywords'])
    
    # Expand query with synonyms
    words = [w for w in q.split() if len(w) >= 2]
    expanded_words = list(words) + extra_keywords
    
    for word in words:
        # Add synonyms
        if word in SYNONYMS:
            expanded_words.extend(SYNONYMS[word])
        
        # Detect category from keywords
        if word in CATEGORY_MAP and not target_category:
            target_category = CATEGORY_MAP[word]["category"]
            target_subcategories.extend(CATEGORY_MAP[word]["subcategories"])
        
        # Detect materials from keywords
        for material, keywords in MATERIAL_KEYWORDS.items():
            if word in keywords:
                target_materials.append(material)
    
    # Remove duplicates
    expanded_words = list(set(expanded_words))
    target_subcategories = list(set(target_subcategories))
    target_materials = list(set(target_materials))
    
    # === STRICT CATEGORY FILTERING ===
    # When user asks for a specific category (shoes, bags, dresses), ONLY show that category
    strict_category_mode = False
    category_keywords_in_query = ['shoes', 'shoe', 'heels', 'heel', 'sandals', 'footwear', 
                                   'bags', 'bag', 'purse', 'clutch', 'handbag',
                                   'dresses', 'dress', 'gown', 'jumpsuits', 'jumpsuit',
                                   'tops', 'top', 'blouse', 'sets', 'set']
    
    for kw in category_keywords_in_query:
        if kw in q:
            strict_category_mode = True
            break
    
    # === FILTER PRODUCTS ===
    filtered_products = products
    
    # Filter by category (STRICT when category explicitly requested)
    if target_category:
        category_filtered = []
        for p in products:
            # Check category field
            if p.get('category', '').lower() == target_category.lower():
                category_filtered.append(p)
            # Fallback: check product_type for category indication
            elif not p.get('category'):
                ptype = p.get('product_type', '').lower()
                if target_category == "Footwear" and any(ft in ptype for ft in ['heel', 'sandal', 'shoe', 'pump', 'mule']):
                    category_filtered.append(p)
                elif target_category == "Accessories" and any(acc in ptype for acc in ['bag', 'clutch', 'purse', 'handbag', 'tote']):
                    category_filtered.append(p)
                elif target_category == "Clothing" and any(cl in ptype for cl in ['dress', 'jumpsuit', 'top', 'blouse', 'set', 'skirt']):
                    category_filtered.append(p)
        
        if category_filtered:
            filtered_products = category_filtered
        elif strict_category_mode:
            # If strict mode and no matches, return empty rather than unrelated products
            return []
    
    # Filter by subcategory
    if target_subcategories:
        subcat_filtered = [
            p for p in filtered_products 
            if any(
                sc.lower() in p.get('subcategory', '').lower() or
                sc.lower() in p.get('product_type', '').lower()
                for sc in target_subcategories
            )
        ]
        if subcat_filtered:
            filtered_products = subcat_filtered
    
    # Filter by material
    if target_materials:
        material_filtered = [
            p for p in filtered_products
            if any(
                mat.lower() in ' '.join(p.get('materials', [])).lower() or
                (mat == 'sequin' and p.get('has_embellishment', False))
                for mat in target_materials
            )
        ]
        if material_filtered:
            filtered_products = material_filtered
    
    # === SCORE PRODUCTS ===
    scored = []
    for p in filtered_products:
        score = 0
        
        # Text fields to search
        name = p.get('product_name', '').lower()
        ptype = p.get('product_type', '').lower()
        subcat = p.get('subcategory', '').lower()
        vibes = ' '.join(p.get('vibe_tags', [])).lower()
        mood = p.get('mood_summary', '').lower()
        ideal = p.get('ideal_for', '').lower()
        materials = ' '.join(p.get('materials', [])).lower()
        occasions = ' '.join(p.get('occasions', [])).lower()
        colors = ' '.join(p.get('colors', [])).lower()
        
        for w in expanded_words:
            w = w.lower()
            if w in name: score += 20
            if w in subcat: score += 18
            if w in ptype: score += 15
            if w in materials: score += 15
            if w in vibes: score += 12
            if w in occasions: score += 10
            if w in mood: score += 8
            if w in ideal: score += 8
            if w in colors: score += 5
        
        # Boost for category/subcategory match
        if target_category and p.get('category', '').lower() == target_category.lower():
            score += 25
        if target_subcategories:
            for sc in target_subcategories:
                if sc.lower() in subcat or sc.lower() in ptype:
                    score += 30
        
        # Boost for material match
        if target_materials:
            for mat in target_materials:
                if mat.lower() in materials:
                    score += 20
                if mat == 'sequin' and p.get('has_embellishment'):
                    score += 20
        
        # Boost for occasion match
        if target_occasions:
            for occ in target_occasions:
                if occ.lower() in occasions or occ.lower() in ideal:
                    score += 15
        
        # In strict category mode, include all filtered products (even with score 0)
        # In non-strict mode, only include if score > 0
        if strict_category_mode:
            # Give minimum score of 1 so all filtered products are included
            scored.append((p, max(score, 1)))
        elif score > 0:
            scored.append((p, score))
    
    # If no scored matches but we have filtered products, return them
    if not scored and filtered_products and (target_category or target_subcategories):
        return filtered_products[:n]
    
    scored.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in scored[:n]]


def find_product(name: str, products: List[Dict]) -> Optional[Dict]:
    if not name:
        return None
    name_lower = name.lower()
    for p in products:
        if name_lower in p.get('product_name', '').lower():
            return p
    return None


def get_stock_info(product: Dict, stock_data: Dict, size: str = None, color: str = None) -> str:
    """Get stock info, optionally for specific size/color"""
    pid = str(product.get('product_id', ''))
    name = product.get('product_name', '')
    
    if pid not in stock_data:
        return "Please contact us for availability of {}.".format(name)
    
    stock = stock_data[pid]
    variants = stock.get('variants', [])
    
    # If specific size/color requested
    if size or color:
        for v in variants:
            v_size = v.get('size', '').upper()
            v_color = v.get('color', '').lower()
            
            size_match = not size or size.upper() in v_size or v_size in size.upper()
            color_match = not color or color.lower() in v_color or v_color in color.lower()
            
            if size_match and color_match:
                qty = v.get('quantity', 0)
                if qty > 0:
                    return "✅ Yes! **{}** in **{}/{}** is available with **{}** in stock.".format(
                        name, v['size'], v['color'], qty
                    )
                else:
                    return "❌ Sorry, **{}** in **{}/{}** is currently out of stock.".format(
                        name, v['size'], v['color']
                    )
        
        return "Sorry, I couldn't find {} in that specific size/color combination.".format(name)
    
    # General stock check
    total = stock.get('total_inventory', 0)
    if total == 0:
        return "Sorry, **{}** is currently out of stock.".format(name)
    
    available = []
    for v in variants:
        if v.get('quantity', 0) > 0:
            available.append("• {}/{}: {} available".format(v['size'], v['color'], v['quantity']))
    
    return "**{}** is in stock!\n\nAvailable:\n{}".format(name, '\n'.join(available[:8]))


def get_policy_answer(query: str, policies: List[Dict]) -> Tuple[str, List[Dict]]:
    policy_rag = get_policy_rag()
    
    if policy_rag:
        openai_client = get_openai_client()
        answer, sections = policy_rag.answer_policy_question(query, openai_client)
        return answer, sections
    
    # Fallback
    q = query.lower()
    if any(w in q for w in ['refund', 'return', 'exchange']):
        policy_type = 'refund_policy'
    elif any(w in q for w in ['ship', 'delivery']):
        policy_type = 'shipping_policy'
    else:
        policy_type = 'terms_of_service'
    
    for policy in policies:
        if policy.get('policy_id') == policy_type:
            return "**{}**\n\n{}...".format(policy['policy_name'], policy['content'][:1500]), []
    
    return "Please contact hello@bynoemie.com for policy questions.", []


# =============================================================================
# PRODUCT DISPLAY
# =============================================================================
def render_products_html(products: List[Dict], stock_data: Dict, images_data: Dict) -> str:
    cards = ""
    
    for p in products:
        pid = str(p.get('product_id', ''))
        name = p.get('product_name', 'Unknown')
        price = p.get('price_min', 0)
        currency = p.get('price_currency', 'MYR')
        ptype = p.get('product_type', '')
        vibes = p.get('vibe_tags', [])[:2]
        
        img_data = images_data.get(pid, {})
        img = img_data.get('image', '') or p.get('image_url_1', '')
        link = img_data.get('link', '') or p.get('product_link', '#')
        
        if not img:
            img = "https://via.placeholder.com/280x350/2a2a2a/d4a5a5?text={}".format(name[:10].replace(' ', '+'))
        
        stock_info = stock_data.get(pid, {})
        total = stock_info.get('total_inventory', 10)
        if total == 0:
            stock_html = '<span style="color:#e57373;">Sold Out</span>'
        elif total <= 5:
            stock_html = '<span style="color:#ffb74d;">Only {} left</span>'.format(total)
        else:
            stock_html = '<span style="color:#81c784;">In Stock</span>'
        
        vibes_html = ''.join(['<span style="background:rgba(212,165,165,0.2);color:#d4a5a5;padding:4px 10px;border-radius:15px;font-size:10px;margin-right:4px;">{}</span>'.format(v) for v in vibes])
        
        cards += '''
        <a href="{link}" target="_blank" style="text-decoration:none;">
            <div style="min-width:200px;max-width:200px;background:#2a2a2a;border-radius:12px;overflow:hidden;flex-shrink:0;margin-right:16px;transition:transform 0.3s;cursor:pointer;"
                 onmouseover="this.style.transform='translateY(-5px)'" 
                 onmouseout="this.style.transform='translateY(0)'">
                <img src="{img}" style="width:100%;height:250px;object-fit:cover;" 
                     onerror="this.src='https://via.placeholder.com/280x350/2a2a2a/d4a5a5?text=No+Image'">
                <div style="padding:12px;">
                    <div style="font-family:'Cormorant Garamond',serif;font-size:15px;font-weight:500;color:#f5f5f5;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{name}</div>
                    <div style="color:#d4a5a5;font-weight:500;font-size:14px;margin:4px 0;">{currency} {price:.0f}</div>
                    <div style="color:#888;font-size:11px;margin-bottom:8px;">{ptype}</div>
                    <div style="margin-bottom:8px;">{vibes_html}</div>
                    <div style="font-size:11px;">{stock_html}</div>
                </div>
            </div>
        </a>'''.format(link=link, img=img, name=name, currency=currency, price=price, ptype=ptype, vibes_html=vibes_html, stock_html=stock_html)
    
    return '<div style="display:flex;overflow-x:auto;padding:16px 0;">{}</div>'.format(cards)


def display_products(products: List[Dict], stock_data: Dict, images_data: Dict, key: str):
    if not products:
        return
    html = render_products_html(products, stock_data, images_data)
    components.html(html, height=420, scrolling=True)


def display_feedback_buttons(key: str, query: str, response: str, product_context: str = None):
    """Display compact feedback buttons with 10px gap"""
    # Use custom HTML container for precise 10px gap
    st.markdown("""
    <style>
    .fb-row-{} {{
        display: flex;
        gap: 10px;
        margin-top: 5px;
    }}
    </style>
    """.format(key), unsafe_allow_html=True)
    
    # Create columns with minimal width
    col1, col2, col3, spacer = st.columns([0.08, 0.08, 0.08, 0.76])
    
    with col1:
        if st.button("👍", key="pos_{}".format(key), help="Helpful"):
            save_feedback("positive", query, response, product_context)
            st.toast("Thank you! 💕")
    
    with col2:
        if st.button("👎", key="neg_{}".format(key), help="Not helpful"):
            save_feedback("negative", query, response, product_context)
            st.toast("We'll improve!")
    
    with col3:
        if st.button("😐", key="neu_{}".format(key), help="Okay"):
            save_feedback("neutral", query, response, product_context)


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    products = load_products()
    stock_data = load_stock()
    images_data = load_images()
    policies = load_policies()
    product_names = [p.get('product_name', '') for p in products]
    order_manager = get_order_manager()
    
    # Header
    st.markdown("""
    <div style="text-align:center;padding:20px 0 30px 0;">
        <h1 style="font-family:'Cormorant Garamond',serif;font-size:42px;font-weight:400;color:#f5f5f5;margin:0;letter-spacing:2px;">BYNOEMIE</h1>
        <p style="font-family:'Montserrat',sans-serif;font-size:12px;color:#888;letter-spacing:3px;margin-top:8px;">FASHION ASSISTANT</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("<h3 style='color:#d4a5a5;'>Quick Actions</h3>", unsafe_allow_html=True)
        
        suggestions = [
            "Show me romantic dinner dresses",
            "Is the Kylie Jumpsuit in stock?",
            "What's your refund policy?",
            "I want to order Luna Dress"
        ]
        
        for s in suggestions:
            if st.button(s, key="sug_{}".format(s[:8]), use_container_width=True):
                st.session_state.auto_query = s
        
        st.markdown("---")
        
        # Current product context
        if st.session_state.get('current_product'):
            st.markdown("<h4 style='color:#d4a5a5;'>Currently Viewing</h4>", unsafe_allow_html=True)
            st.write("**{}**".format(st.session_state.current_product))
            if st.button("Clear", use_container_width=True):
                st.session_state.current_product = None
        
        st.markdown("---")
        st.metric("Products", len(products))
        
        if st.button("🚪 End Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_product = None
            st.rerun()
    
    if not products:
        st.error("Unable to load products.")
        return
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_product" not in st.session_state:
        st.session_state.current_product = None
    if "current_order" not in st.session_state:
        st.session_state.current_order = None
    
    # Display chat history
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("products"):
                display_products(msg["products"], stock_data, images_data, "h{}".format(i))
            if msg["role"] == "assistant":
                display_feedback_buttons(
                    "hist_{}".format(i),
                    msg.get("query", ""),
                    msg["content"],
                    st.session_state.current_product
                )
    
    # Input
    query = st.session_state.pop("auto_query", None) or st.chat_input("How may I assist you?")
    
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # Classify intent WITH FULL CHAT HISTORY (OpenAI)
        intent_result = llm_classify_intent(
            query, 
            product_names,
            chat_history=st.session_state.messages[:-1],  # All messages except current
            current_product=st.session_state.current_product
        )
        intent = intent_result.get("intent", "GENERAL")
        product_mentioned = intent_result.get("product_mentioned")
        size_mentioned = intent_result.get("size")
        color_mentioned = intent_result.get("color")
        
        with st.chat_message("assistant"):
            response = ""
            show_products = []
            response_placeholder = st.empty()
            
            # Get product
            product = find_product(product_mentioned, products) if product_mentioned else None
            
            # Update context
            if product:
                st.session_state.current_product = product.get('product_name')
            
            client = get_openai_client()
            
            # === STOCK CHECK ===
            if intent == "STOCK_CHECK":
                if product:
                    fresh_stock = reload_stock()
                    stock_info = get_stock_info(product, fresh_stock, size_mentioned, color_mentioned)
                    
                    if client:
                        messages = [
                            {"role": "system", "content": "Provide stock info professionally."},
                            {"role": "user", "content": "Query: {}\nStock Info: {}".format(query, stock_info)}
                        ]
                        response = stream_response(client, messages, response_placeholder) or stock_info
                    else:
                        response = stock_info
                        response_placeholder.markdown(response)
                    show_products = [product]
                else:
                    response = "Which product would you like me to check? Please mention the name."
                    response_placeholder.markdown(response)
            
            # === ORDER CONFIRM (User types ORDER/DELETE/CHANGE) ===
            elif intent == "ORDER_CONFIRM":
                confirm_type = intent_result.get('confirm_type', '')
                pending = st.session_state.get('pending_order_action')
                
                if confirm_type == "ORDER" and pending and pending.get('action') == 'create':
                    # Complete order creation
                    order_manager = get_order_manager()
                    if order_manager:
                        order_data = pending.get('data', {})
                        user_id = st.session_state.get('current_user_id', 'USR-001')
                        
                        order = order_manager.create_order(
                            user_id=user_id,
                            product=order_data.get('product', {}),
                            size=order_data.get('size', 'M'),
                            color=order_data.get('color', 'Default'),
                            quantity=order_data.get('quantity', 1)
                        )
                        
                        response = f"""✅ **Order Confirmed!**

Your order has been placed successfully!

{order_manager.format_order_summary(order)}

📧 You will receive a confirmation email shortly.
📦 Estimated delivery: {order['estimated_delivery']}

Thank you for shopping with ByNoemie! 💕"""
                        st.session_state.pending_order_action = None
                    else:
                        response = "Sorry, order system is temporarily unavailable."
                    response_placeholder.markdown(response)
                    
                elif confirm_type == "DELETE" and pending and pending.get('action') == 'cancel':
                    # Complete order cancellation
                    order_manager = get_order_manager()
                    if order_manager:
                        order_id = pending.get('order_id')
                        success, message, order = order_manager.cancel_order(order_id)
                        
                        if success:
                            response = f"""✅ **Order Cancelled**

{message}

Your refund will be processed within 3-5 business days.

Is there anything else I can help you with?"""
                        else:
                            response = f"❌ {message}"
                    else:
                        response = "Sorry, order system is temporarily unavailable."
                    response_placeholder.markdown(response)
                    st.session_state.pending_order_action = None
                    
                elif confirm_type == "CHANGE" and pending and pending.get('action') == 'modify':
                    # Complete order modification
                    order_manager = get_order_manager()
                    if order_manager:
                        order_id = pending.get('order_id')
                        changes = pending.get('changes', {})
                        
                        success, message, order = order_manager.modify_order(
                            order_id,
                            new_size=changes.get('size'),
                            new_color=changes.get('color'),
                            new_quantity=changes.get('quantity')
                        )
                        
                        if success:
                            response = f"""✅ **Order Modified**

Changes applied:
{message}

**Updated Order:**
{order_manager.format_order_summary(order)}

Is there anything else I can help you with?"""
                        else:
                            response = f"❌ {message}"
                    else:
                        response = "Sorry, order system is temporarily unavailable."
                    response_placeholder.markdown(response)
                    st.session_state.pending_order_action = None
                else:
                    response = "I don't have a pending action to confirm. How can I help you?"
                    response_placeholder.markdown(response)
            
            # === ORDER CREATE ===
            elif intent == "ORDER_CREATE":
                order_manager = get_order_manager()
                
                if product and order_manager:
                    size_to_order = size_mentioned or 'M'
                    color_to_order = color_mentioned or product.get('colors_available', 'Default').split(',')[0].strip()
                    
                    # Store pending order
                    st.session_state.pending_order_action = {
                        'action': 'create',
                        'data': {
                            'product': product,
                            'size': size_to_order,
                            'color': color_to_order,
                            'quantity': 1
                        }
                    }
                    
                    response = f"""📝 **Order Summary**

• **Product:** {product.get('product_name')}
• **Size:** {size_to_order}
• **Color:** {color_to_order}
• **Quantity:** 1
• **Price:** {product.get('price_currency', 'MYR')} {product.get('price_min', 0):.2f}

⚠️ **To confirm your order, please type:** `ORDER`

_Type anything else to cancel._"""
                    response_placeholder.markdown(response)
                    show_products = [product]
                else:
                    response = """I'd love to help you place an order! 

Could you please specify:
• Which product you'd like to order
• Size preference
• Color preference

For example: "I want to order the Luna Dress in size M, White" """
                    response_placeholder.markdown(response)
            
            # === ORDER CANCEL ===
            elif intent == "ORDER_CANCEL":
                order_manager = get_order_manager()
                order_id = intent_result.get('order_id')
                
                if order_manager:
                    if order_id:
                        can_cancel, reason = order_manager.can_cancel_order(order_id)
                        order = order_manager.get_order(order_id)
                        
                        if can_cancel and order:
                            st.session_state.pending_order_action = {
                                'action': 'cancel',
                                'order_id': order_id
                            }
                            
                            response = f"""🗑️ **Cancel Order Request**

{order_manager.format_order_summary(order)}

⚠️ **Are you sure you want to cancel this order?**

**To confirm cancellation, please type:** `DELETE`

_Type anything else to keep the order._"""
                        else:
                            response = f"❌ {reason}"
                        response_placeholder.markdown(response)
                    else:
                        recent = order_manager.get_recent_orders(5)
                        if recent:
                            orders_list = '\n'.join([
                                f"• **{o['order_id']}** - {o['product_name']} ({o['status'].replace('_', ' ').title()})"
                                for o in recent
                            ])
                            response = f"""Which order would you like to cancel?

**Your Recent Orders:**
{orders_list}

Please specify the order ID (e.g., "Cancel order ORD-001")"""
                        else:
                            response = "You don't have any orders yet."
                        response_placeholder.markdown(response)
                else:
                    response = "Sorry, order system is temporarily unavailable."
                    response_placeholder.markdown(response)
            
            # === ORDER MODIFY ===
            elif intent == "ORDER_MODIFY":
                order_manager = get_order_manager()
                order_id = intent_result.get('order_id')
                
                if order_manager:
                    if order_id:
                        can_modify, reason = order_manager.can_modify_order(order_id)
                        order = order_manager.get_order(order_id)
                        
                        if can_modify and order:
                            new_size = size_mentioned
                            new_color = color_mentioned
                            
                            if new_size or new_color:
                                st.session_state.pending_order_action = {
                                    'action': 'modify',
                                    'order_id': order_id,
                                    'changes': {'size': new_size, 'color': new_color}
                                }
                                
                                changes_text = []
                                if new_size:
                                    changes_text.append(f"Size: {order['size']} → {new_size}")
                                if new_color:
                                    changes_text.append(f"Color: {order['color']} → {new_color}")
                                
                                response = f"""✏️ **Modify Order Request**

**Order:** {order_id}
**Product:** {order['product_name']}

**Requested Changes:**
• {chr(10).join(['• ' + c for c in changes_text[1:]]) if len(changes_text) > 1 else changes_text[0] if changes_text else 'None'}

⚠️ **To confirm changes, please type:** `CHANGE`

_Type anything else to cancel._"""
                            else:
                                response = f"""What would you like to change for order {order_id}?

**Current Order:**
{order_manager.format_order_summary(order)}

You can change: Size, Color, or Quantity
Example: "Change order {order_id} to size L" """
                        else:
                            response = f"❌ {reason}"
                        response_placeholder.markdown(response)
                    else:
                        recent = order_manager.get_recent_orders(5)
                        modifiable = [o for o in recent if o['status'] in ['pending_confirmation', 'confirmed', 'processing']]
                        
                        if modifiable:
                            orders_list = '\n'.join([
                                f"• **{o['order_id']}** - {o['product_name']} (Size: {o['size']}, Color: {o['color']})"
                                for o in modifiable
                            ])
                            response = f"""Which order would you like to modify?

**Orders that can be modified:**
{orders_list}

Example: "Change order ORD-002 to size L" """
                        else:
                            response = "You don't have any orders that can be modified. Orders that have been shipped cannot be changed."
                        response_placeholder.markdown(response)
                else:
                    response = "Sorry, order system is temporarily unavailable."
                    response_placeholder.markdown(response)
            
            # === ORDER TRACK ===
            elif intent == "ORDER_TRACK":
                order_manager = get_order_manager()
                order_id = intent_result.get('order_id')
                
                if order_manager:
                    if order_id:
                        response = order_manager.track_order(order_id)
                    else:
                        recent = order_manager.get_recent_orders(5)
                        if recent:
                            response = "📦 **Your Recent Orders:**\n\n"
                            for order in recent:
                                status_emoji = {"pending_confirmation": "⏳", "confirmed": "✅", 
                                               "processing": "📋", "shipped": "🚚", 
                                               "delivered": "🎉", "cancelled": "❌"}
                                emoji = status_emoji.get(order['status'], "📦")
                                response += f"{emoji} **{order['order_id']}** - {order['product_name']}\n"
                                response += f"   Status: {order['status'].replace('_', ' ').title()}\n\n"
                            response += "_Say 'Track order ORD-XXX' for details._"
                        else:
                            response = "You don't have any orders yet. Would you like to browse our collection?"
                    response_placeholder.markdown(response)
                else:
                    response = "Sorry, order system is temporarily unavailable."
                    response_placeholder.markdown(response)
            
            # === USER PROFILE ===
            elif intent == "USER_PROFILE":
                user_manager = get_user_manager()
                user_id = st.session_state.get('current_user_id', 'USR-001')
                
                if user_manager:
                    user = user_manager.get_user(user_id)
                    if user:
                        response = user_manager.format_user_profile(user)
                    else:
                        response = "User profile not found. Please contact customer service."
                else:
                    response = "Sorry, profile system is temporarily unavailable."
                response_placeholder.markdown(response)
            
            # === RECOMMENDATION ===
            elif intent == "RECOMMENDATION":
                # Use LLM to understand what user wants
                query_understanding = llm_understand_query(query, client) if client else {}
                
                # Detect if this is a broad category query
                broad_categories = {
                    "shoes": "Footwear",
                    "shoe": "Footwear",
                    "heels": "Footwear",
                    "heel": "Footwear",
                    "footwear": "Footwear",
                    "sandals": "Footwear",
                    "bags": "Accessories",
                    "bag": "Accessories",
                    "purse": "Accessories",
                    "clutch": "Accessories",
                    "dresses": "Clothing",
                    "dress": "Clothing",
                    "jumpsuits": "Clothing",
                    "jumpsuit": "Clothing",
                    "tops": "Clothing",
                    "top": "Clothing",
                    "sets": "Clothing",
                    "set": "Clothing",
                    "clothing": "Clothing",
                    "clothes": "Clothing",
                }
                
                q_lower = query.lower()
                is_broad_query = False
                category_requested = None
                
                # Check if query is asking for a broad category
                for cat_keyword, cat_name in broad_categories.items():
                    if cat_keyword in q_lower and not any(specific in q_lower for specific in ['for wedding', 'for party', 'for dinner', 'sparkly', 'silk', 'sequin', 'black', 'white', 'red', 'gold']):
                        is_broad_query = True
                        category_requested = cat_name
                        break
                
                # Search with understanding - use 10 products for broad queries
                max_results = 10 if is_broad_query else 8
                results = search_products(query, products, max_results, query_understanding)
                
                if results:
                    # Get product types found for better response
                    types_found = list(set([p.get('product_type', '') for p in results]))
                    subcats_found = list(set([p.get('subcategory', '') for p in results if p.get('subcategory')]))
                    ctx = ', '.join([p.get('product_name', '') for p in results[:5]])
                    
                    if is_broad_query and client:
                        # For broad queries, show products AND ask follow-up
                        follow_up_prompt = """You're a fashion stylist. The customer asked for "{}" and we found {} options.

Products found: {}

Write a SHORT response (2-3 sentences) that:
1. Shows enthusiasm about the options
2. Mentions 1-2 specific product names
3. Asks ONE helpful follow-up question about:
   - What occasion they're shopping for (party, work, dinner, wedding)
   - OR what style they prefer (elegant, bold, minimalist)
   - OR what heel height/comfort level they want (for shoes)

Be conversational, not robotic.""".format(query, len(results), ctx)
                        
                        messages = [
                            {"role": "system", "content": follow_up_prompt},
                            {"role": "user", "content": query}
                        ]
                        response = stream_response(client, messages, response_placeholder)
                    elif client:
                        messages = [
                            {"role": "system", "content": "Fashion stylist. SHORT rec (2 sentences). Mention specific product names from the list."},
                            {"role": "user", "content": "Query: {}\nProducts found ({}): {}".format(query, ', '.join(types_found), ctx)}
                        ]
                        response = stream_response(client, messages, response_placeholder)
                    
                    if not response:
                        if is_broad_query:
                            response = "Here are {} {} options for you! ✨ What occasion are you shopping for?".format(len(results), types_found[0] if types_found else 'pieces')
                        else:
                            response = "Here are {} {} for you! ✨".format(len(results), types_found[0] if types_found else 'pieces')
                        response_placeholder.markdown(response)
                    show_products = results
                    st.session_state.current_product = results[0].get('product_name')
                else:
                    response = "I couldn't find products matching that. Could you try different keywords or browse our collection?"
                    response_placeholder.markdown(response)
            
            # === PRODUCT INFO ===
            elif intent == "PRODUCT_INFO":
                if product:
                    # Detect what specific info user is asking about
                    q_lower = query.lower()
                    
                    # Extract product attributes
                    colors = product.get('colors_available', 'Not specified')
                    sizes = product.get('size_options', 'Not specified')
                    price = "{} {}".format(product.get('price_currency', 'MYR'), product.get('price_min', 0))
                    materials = ', '.join(product.get('materials', [])) or product.get('material', 'Not specified')
                    name = product.get('product_name', '')
                    description = product.get('product_description', '')
                    
                    # Determine what user is asking about
                    specific_answer = None
                    
                    if any(w in q_lower for w in ['colour', 'color', 'colours', 'colors']):
                        specific_answer = "The {} comes in: **{}**".format(name, colors)
                    elif any(w in q_lower for w in ['size', 'sizes', 'fit', 'fitting']):
                        specific_answer = "The {} is available in: **{}**".format(name, sizes)
                    elif any(w in q_lower for w in ['price', 'cost', 'how much']):
                        specific_answer = "The {} is priced at **{}**".format(name, price)
                    elif any(w in q_lower for w in ['material', 'fabric', 'made of', 'made from']):
                        specific_answer = "The {} is made of: **{}**".format(name, materials)
                    elif any(w in q_lower for w in ['available', 'in stock', 'stock']):
                        fresh_stock = reload_stock()
                        stock_info = get_stock_info(product, fresh_stock)
                        specific_answer = stock_info
                    
                    if specific_answer:
                        # Answer the specific question directly
                        if client:
                            messages = [
                                {"role": "system", "content": """You are a helpful fashion assistant. 
Answer the customer's specific question FIRST in 1 sentence using the info provided.
Then optionally add ONE brief styling tip or suggestion (1 sentence max).
Keep it SHORT and DIRECT. No bullet points."""},
                                {"role": "user", "content": "Customer asked: {}\nAnswer: {}\nProduct: {}".format(query, specific_answer, name)}
                            ]
                            response = stream_response(client, messages, response_placeholder)
                        if not response:
                            response = specific_answer
                            response_placeholder.markdown(response)
                    else:
                        # General product info request
                        info = "{} - {}\n{}\nColors: {} | Sizes: {}".format(
                            name, price, 
                            product.get('mood_summary', description[:200]),
                            colors, sizes
                        )
                        if client:
                            messages = [
                                {"role": "system", "content": "Describe product in 2-3 sentences. Mention key details. Be concise."},
                                {"role": "user", "content": info}
                            ]
                            response = stream_response(client, messages, response_placeholder) or info
                        else:
                            response = info
                            response_placeholder.markdown(response)
                    show_products = [product]
                else:
                    # Try to find product from context or search
                    if st.session_state.get('current_product'):
                        product = find_product(st.session_state.current_product, products)
                        if product:
                            # Re-process with found product
                            st.session_state.messages.append({"role": "user", "content": query})
                            st.rerun()
                    response = "Which product would you like to know about? Please mention the name."
                    response_placeholder.markdown(response)
            
            # === POLICY ===
            elif intent == "POLICY_QUESTION":
                policy_answer, _ = get_policy_answer(query, policies)
                if client:
                    messages = [
                        {"role": "system", "content": "Answer policy question helpfully."},
                        {"role": "user", "content": "Query: {}\nPolicy: {}".format(query, policy_answer[:1000])}
                    ]
                    response = stream_response(client, messages, response_placeholder) or policy_answer
                else:
                    response = policy_answer
                    response_placeholder.markdown(response)
            
            # === GREETING ===
            elif intent == "GREETING":
                if client:
                    messages = [
                        {"role": "system", "content": "Warm fashion assistant. Brief greeting."},
                        {"role": "user", "content": query}
                    ]
                    response = stream_response(client, messages, response_placeholder)
                if not response:
                    response = "Welcome to ByNoemie! ✨ How can I help you today?"
                    response_placeholder.markdown(response)
            
            # === END ===
            elif intent == "END_CONVERSATION":
                response = "Thank you for visiting ByNoemie! 💕 Have a wonderful day!"
                response_placeholder.markdown(response)
            
            # === THANKS ===
            elif intent == "THANKS":
                response = "You're welcome! 💕 Anything else I can help with?"
                response_placeholder.markdown(response)
            
            # === OFF_TOPIC - Politely redirect to fashion ===
            elif intent == "OFF_TOPIC":
                if client:
                    # Use LLM to generate varied, natural redirect responses
                    redirect_prompt = """You are ByNoemie's friendly fashion assistant. The user asked something off-topic (not about fashion/products/store).

User's message: "{}"

Respond naturally and conversationally (1-2 sentences). Be warm but brief. Politely mention you're here to help with fashion/shopping, then ask what they're looking for OR suggest browsing dresses/heels/bags.

VARY your responses. Examples of good responses:
- "Haha, I wish I could help with that! But I'm here for all things fashion 👗 Looking for something special today?"
- "That's outside my expertise! I'm your fashion guide - want to explore our new arrivals?"
- "I'll leave that to Google 😄 But if you need outfit advice, I'm your girl! Any occasion coming up?"
- "Not my area, but I do know style! Need help finding the perfect dress or heels?"

Do NOT use bullet points. Keep it casual and short.""".format(query)
                    
                    messages = [
                        {"role": "system", "content": redirect_prompt},
                        {"role": "user", "content": query}
                    ]
                    response = stream_response(client, messages, response_placeholder)
                
                if not response:
                    # Fallback varied responses
                    import random
                    fallbacks = [
                        "I'm all about fashion! 👗 Can I help you find something fabulous to wear?",
                        "That's a bit outside my wheelhouse! But I'd love to help you find the perfect outfit - what's the occasion?",
                        "I'm your style assistant, so fashion is my thing! Looking for dresses, heels, or something else?",
                        "Hmm, I'm better at outfit advice! 😊 Want me to show you our latest pieces?",
                        "I stick to the fashion lane! Need help finding something gorgeous?",
                    ]
                    response = random.choice(fallbacks)
                    response_placeholder.markdown(response)
            
            # === DEFAULT (GENERAL fashion questions) ===
            else:
                if client:
                    messages = [
                        {"role": "system", "content": "You are ByNoemie's fashion assistant. ONLY answer questions about fashion, clothing, style, and our store. If the question is not about fashion/clothing/store, politely say you can only help with fashion-related questions."},
                        {"role": "user", "content": query}
                    ]
                    response = stream_response(client, messages, response_placeholder)
                if not response:
                    response = "I can help with recommendations, stock checks, and policy questions!"
                    response_placeholder.markdown(response)
            
            # Show products
            if show_products:
                display_products(show_products, stock_data, images_data, "n{}".format(len(st.session_state.messages)))
            
            # Feedback for every response
            display_feedback_buttons(
                "new_{}".format(len(st.session_state.messages)),
                query,
                response or "Response",
                st.session_state.current_product
            )
            
            # Save message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response or "Response",
                "products": show_products,
                "query": query
            })


if __name__ == "__main__":
    main()