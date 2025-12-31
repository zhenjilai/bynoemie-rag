"""
ByNoemie RAG Chatbot v2 - Multi-Agent Architecture

Agents:
- RouterAgent: Routes queries to appropriate specialist
- DeflectionAgent: Handles off-topic, greetings
- InfoAgent: Product info, recommendations, stock, policy, tracking
- ActionAgent: Order create/modify/cancel with validation
- ConfirmationAgent: Handles ORDER/DELETE/CHANGE

Usage:
    streamlit run app_v2.py
"""

import os
import json
import streamlit as st
import streamlit.components.v1 as components
from typing import List, Dict, Optional
from datetime import datetime
import uuid
import re

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

h1, h2, h3, h4 {
    color: #d4a574 !important;
    font-family: 'Cormorant Garamond', serif !important;
}

p, span, li, div {
    font-family: 'Montserrat', sans-serif !important;
}

.product-card {
    background: linear-gradient(145deg, #2a2a2a 0%, #1f1f1f 100%);
    border-radius: 12px;
    padding: 12px;
    margin: 4px 0;
    border: 1px solid #3a3a3a;
    transition: all 0.3s ease;
    cursor: pointer;
}

.product-card:hover {
    border-color: #d4a574;
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(212, 165, 116, 0.15);
}

/* Horizontal scroll container */
.products-scroll-container {
    display: flex;
    overflow-x: auto;
    gap: 16px;
    padding: 10px 0 20px 0;
    scroll-behavior: smooth;
    -webkit-overflow-scrolling: touch;
}

.products-scroll-container::-webkit-scrollbar {
    height: 6px;
}

.products-scroll-container::-webkit-scrollbar-track {
    background: #1a1a1a;
    border-radius: 3px;
}

.products-scroll-container::-webkit-scrollbar-thumb {
    background: #3a3a3a;
    border-radius: 3px;
}

.products-scroll-container::-webkit-scrollbar-thumb:hover {
    background: #d4a574;
}

.product-card-link {
    text-decoration: none;
    flex-shrink: 0;
}

.product-card-scroll {
    width: 180px;
    background: linear-gradient(145deg, #2a2a2a 0%, #1f1f1f 100%);
    border-radius: 12px;
    border: 1px solid #3a3a3a;
    overflow: hidden;
    transition: all 0.3s ease;
    cursor: pointer;
}

.product-card-scroll:hover {
    border-color: #d4a574;
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(212, 165, 116, 0.15);
}

.product-image-container {
    width: 100%;
    height: 200px;
    overflow: hidden;
    background: #1a1a1a;
}

.product-img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 0.3s ease;
}

.product-card-scroll:hover .product-img {
    transform: scale(1.05);
}

.product-details {
    padding: 12px;
}

.product-name {
    color: #f5f5f5;
    font-family: 'Montserrat', sans-serif;
    font-size: 0.9em;
    font-weight: 500;
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.product-price {
    color: #d4a574;
    font-family: 'Montserrat', sans-serif;
    font-size: 0.95em;
    font-weight: 600;
    margin-bottom: 4px;
}

.product-category {
    color: #888;
    font-size: 0.75em;
    margin-bottom: 6px;
}

.product-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-bottom: 8px;
}

.product-tag {
    background: rgba(212, 165, 116, 0.1);
    border: 1px solid rgba(212, 165, 116, 0.3);
    color: #d4a574;
    font-size: 0.65em;
    padding: 2px 6px;
    border-radius: 10px;
    white-space: nowrap;
}

.stock-in {
    color: #2ecc71;
    font-size: 0.75em;
    font-weight: 500;
}

.stock-low {
    color: #e74c3c;
    font-size: 0.75em;
    font-weight: 500;
}

.stock-out {
    color: #888;
    font-size: 0.75em;
    font-weight: 500;
}

.stock-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.7em;
    font-weight: 500;
    margin-top: 4px;
}

.in-stock {
    background: rgba(46, 204, 113, 0.2);
    color: #2ecc71;
}

.low-stock {
    background: rgba(241, 196, 15, 0.2);
    color: #f1c40f;
}

.product-image {
    max-height: 180px;
    object-fit: cover;
    border-radius: 8px;
    margin-top: 8px;
}

/* Compact product action buttons */
.stButton > button {
    font-size: 0.8em !important;
    padding: 4px 8px !important;
}

.feedback-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 8px 16px;
    border-radius: 20px;
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    color: #888;
    cursor: pointer;
    transition: all 0.2s ease;
    margin: 4px;
}

.feedback-btn:hover {
    border-color: #d4a574;
    color: #d4a574;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def load_products():
    paths = ["data/products/bynoemie_products.json", "bynoemie_products.json"]
    for path in paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    return []


@st.cache_data(ttl=60, show_spinner=False)
def load_stock():
    paths = ["data/stock/bynoemie_stock.json", "bynoemie_stock.json"]
    for path in paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    return {}


@st.cache_data(ttl=300, show_spinner=False)
def load_images():
    paths = ["data/products/bynoemie_images.json", "bynoemie_images.json"]
    for path in paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    return {}


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


def reload_stock():
    load_stock.clear()
    return load_stock()


# =============================================================================
# CLIENT INITIALIZATION
# =============================================================================
def get_openai_client():
    if 'openai_client' not in st.session_state:
        api_key = get_api_key("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI
                st.session_state.openai_client = OpenAI(api_key=api_key)
            except:
                st.session_state.openai_client = None
        else:
            st.session_state.openai_client = None
    return st.session_state.openai_client


def get_order_manager():
    if 'order_manager' not in st.session_state:
        try:
            from src.data_manager import OrderManager, UserManager
            user_mgr = UserManager()
            st.session_state.order_manager = OrderManager(user_mgr)
            st.session_state.user_manager = user_mgr
        except Exception as e:
            print(f"OrderManager init error: {e}")
            st.session_state.order_manager = None
            st.session_state.user_manager = None
    return st.session_state.order_manager


def get_user_manager():
    if 'user_manager' not in st.session_state:
        get_order_manager()  # This initializes both
    return st.session_state.get('user_manager')


def get_policy_rag():
    if 'policy_rag' not in st.session_state:
        try:
            from src.policy_rag import PolicyRAG
            st.session_state.policy_rag = PolicyRAG()
        except:
            st.session_state.policy_rag = None
    return st.session_state.policy_rag


# =============================================================================
# CHATBOT ORCHESTRATOR
# =============================================================================
def get_orchestrator():
    """Get or create the chatbot orchestrator"""
    if 'orchestrator' not in st.session_state:
        try:
            from src.agents import ChatbotOrchestrator
            
            products = load_products()
            stock_data = load_stock()
            openai_client = get_openai_client()
            order_manager = get_order_manager()
            user_manager = get_user_manager()
            policy_rag = get_policy_rag()
            
            st.session_state.orchestrator = ChatbotOrchestrator(
                openai_client=openai_client,
                products=products,
                stock_data=stock_data,
                order_manager=order_manager,
                user_manager=user_manager,
                policy_rag=policy_rag
            )
        except Exception as e:
            print(f"Orchestrator init error: {e}")
            st.session_state.orchestrator = None
    return st.session_state.orchestrator


# =============================================================================
# FEEDBACK SYSTEM
# =============================================================================
def save_feedback(feedback_type: str, query: str, response: str, 
                  product_context: str = None, message_id: str = None):
    """Save feedback to JSON file"""
    feedback_id = message_id or f"fb_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
    feedback_data = {
        "feedback_id": feedback_id,
        "type": feedback_type,
        "query": query,
        "response": response[:500],
        "product_context": product_context,
        "timestamp": datetime.now().isoformat()
    }
    
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


def display_feedback_buttons(message_id: str, query: str, response: str, product_context: str = None):
    """Display feedback buttons"""
    col1, col2, col3, col4 = st.columns([1, 1, 1, 6])
    
    with col1:
        if st.button("👍", key=f"like_{message_id}"):
            save_feedback("positive", query, response, product_context, message_id)
            st.toast("Thanks for the feedback! 💕")
    
    with col2:
        if st.button("👎", key=f"dislike_{message_id}"):
            save_feedback("negative", query, response, product_context, message_id)
            st.toast("Thanks! We'll improve. 🙏")
    
    with col3:
        if st.button("😐", key=f"neutral_{message_id}"):
            save_feedback("neutral", query, response, product_context, message_id)
            st.toast("Noted! 📝")


# =============================================================================
# PRODUCT DISPLAY - Horizontal Scroll Style
# =============================================================================
def display_products(products: List[Dict], stock_data: Dict, images_data: Dict, key_prefix: str = ""):
    """Display products in horizontal scrollable row with tags"""
    if not products:
        return
    
    # Build HTML for horizontal scroll
    cards_html = ""
    for i, product in enumerate(products[:8]):  # Show up to 8 products
        product_handle = product.get('product_handle', '')
        product_url = product.get('product_link', f"https://bynoemie.com.my/products/{product_handle}")
        
        # Get image
        image_url = None
        if product_handle in images_data:
            image_url = images_data[product_handle].get('image_1')
        if not image_url:
            image_url = product.get('image_url_1', '')
        
        # Stock status
        stock_status = product.get('stock_availability', 'In Stock')
        total_inventory = product.get('total_inventory', 0)
        
        if stock_status == 'In Stock':
            if total_inventory and 0 < total_inventory <= 5:
                stock_html = f'<div class="stock-low">Only {total_inventory} left</div>'
            else:
                stock_html = '<div class="stock-in">In Stock</div>'
        else:
            stock_html = '<div class="stock-out">Out of Stock</div>'
        
        # Category
        category = product.get('subcategory', '') or product.get('product_type', '')
        category_html = f'<div class="product-category">{category}</div>' if category else ''
        
        # Tags (vibe_tags or style_attributes)
        tags = product.get('vibe_tags', []) or product.get('style_attributes', [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',')]
        tags_html = ''.join([f'<span class="product-tag">{tag}</span>' for tag in tags[:3]])
        
        # Build card
        cards_html += f'''
        <a href="{product_url}" target="_blank" class="product-card-link">
            <div class="product-card-scroll">
                <div class="product-image-container">
                    <img src="{image_url}" class="product-img" onerror="this.style.display='none'">
                </div>
                <div class="product-details">
                    <div class="product-name">{product.get('product_name', 'Product')}</div>
                    <div class="product-price">MYR {product.get('price_min', 0):.0f}</div>
                    {category_html}
                    <div class="product-tags">{tags_html}</div>
                    {stock_html}
                </div>
            </div>
        </a>
        '''
    
    # Wrap in scrollable container
    st.markdown(f'''
    <div class="products-scroll-container">
        {cards_html}
    </div>
    ''', unsafe_allow_html=True)


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 2.5em; margin-bottom: 5px;">✨ ByNoemie ✨</h1>
        <p style="color: #888; font-size: 1.1em;">Your Personal Fashion Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    products = load_products()
    stock_data = load_stock()
    images_data = load_images()
    
    # Initialize orchestrator
    orchestrator = get_orchestrator()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show products if any
            if msg.get("products"):
                display_products(msg["products"], stock_data, images_data, f"hist_{i}")
            
            # Feedback buttons for assistant messages
            if msg["role"] == "assistant":
                display_feedback_buttons(
                    f"hist_{i}",
                    msg.get("query", ""),
                    msg["content"],
                    orchestrator.state.current_product if orchestrator else None
                )
    
    # Chat input
    query = st.session_state.pop("auto_query", None) or st.chat_input("How may I assist you?")
    
    if query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # Process with orchestrator
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            if orchestrator:
                # Process query through multi-agent system
                # PASS CHAT HISTORY so agents have full context!
                agent_response = orchestrator.process(
                    query, 
                    chat_history=st.session_state.messages  # Pass full history
                )
                
                response = agent_response.message
                products_to_show = agent_response.products_to_show
                
                # Display response
                response_placeholder.markdown(response)
                
                # Display products
                if products_to_show:
                    display_products(products_to_show, stock_data, images_data, f"new_{len(st.session_state.messages)}")
                
                # Feedback buttons
                display_feedback_buttons(
                    f"new_{len(st.session_state.messages)}",
                    query,
                    response,
                    orchestrator.state.current_product
                )
                
                # Save to session
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "products": products_to_show,
                    "query": query
                })
            else:
                # Fallback if orchestrator not available
                response = "I'm having trouble connecting to my systems. Please try again in a moment."
                response_placeholder.markdown(response)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "query": query
                })
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🛍️ Quick Actions")
        
        if st.button("🏠 New Chat", use_container_width=True):
            st.session_state.messages = []
            if orchestrator:
                orchestrator.clear_state()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📦 Browse")
        
        if st.button("👗 Dresses", use_container_width=True):
            st.session_state.auto_query = "Show me dresses"
            st.rerun()
        
        if st.button("👠 Heels", use_container_width=True):
            st.session_state.auto_query = "Show me heels"
            st.rerun()
        
        if st.button("👜 Bags", use_container_width=True):
            st.session_state.auto_query = "Show me bags"
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📋 Orders")
        
        if st.button("📦 My Orders", use_container_width=True):
            st.session_state.auto_query = "Check my orders"
            st.rerun()
        
        if st.button("👤 My Profile", use_container_width=True):
            st.session_state.auto_query = "Show my profile"
            st.rerun()
        
        # Debug info
        if st.checkbox("🔧 Debug"):
            if orchestrator:
                state = orchestrator.get_state()
                st.write("**Current Product:**", state.current_product)
                st.write("**Pending Action:**", state.pending_action)
                st.write("**User ID:**", state.current_user_id)
                st.write("**History Length:**", len(state.conversation_history))


if __name__ == "__main__":
    main()