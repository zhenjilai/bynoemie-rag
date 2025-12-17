"""
ByNoemie RAG Chatbot - Streamlit Demo App

Free deployment on:
- Streamlit Cloud (recommended)
- Hugging Face Spaces
- Railway
- Render

Usage:
    streamlit run app.py
"""

import os
import json
import streamlit as st
from typing import List, Dict, Any

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="ByNoemie Fashion Assistant",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# LOAD ENVIRONMENT (for cloud deployment)
# =============================================================================
# Streamlit Cloud: Use st.secrets
# Other platforms: Use environment variables

def get_api_key(key_name: str) -> str:
    """Get API key from secrets or environment"""
    # Try Streamlit secrets first
    try:
        if hasattr(st, 'secrets') and key_name in st.secrets:
            return st.secrets[key_name]
    except:
        pass
    
    # Fall back to environment variables
    return os.getenv(key_name, "")


# =============================================================================
# SAMPLE DATA (for demo without database)
# =============================================================================
SAMPLE_PRODUCTS = [
    {
        "product_id": "PROD001",
        "product_name": "Coco Dress",
        "product_type": "Dress",
        "product_description": "All eyes on you in the Coco Dress, an ultra-mini silhouette covered in oversized black sequins. Featuring slim straps and a daring open back, it's made for nights that sparkle.",
        "colors_available": "Black, Gold",
        "material": "Sequin",
        "price": 389,
        "vibe_tags": ["main character energy", "disco diva", "NYE countdown ready", "dance floor queen", "night out", "glamorous", "bold"],
        "mood_summary": "For the woman who doesn't just enter a room - she commands it."
    },
    {
        "product_id": "PROD002",
        "product_name": "Tiara Satin Dress",
        "product_type": "Dress",
        "product_description": "Feminine with a touch of fantasy. Features a silky draped neckline, sheer embellished bust detail, and open back with gathered straps.",
        "colors_available": "White, Champagne",
        "material": "Silk, Satin",
        "price": 459,
        "vibe_tags": ["modern fairytale", "romantic dreamer", "ethereal goddess", "bridal shower ready", "soft sensuality", "elegant"],
        "mood_summary": "Where dreams meet reality in a cascade of silk and light."
    },
    {
        "product_id": "PROD003",
        "product_name": "Luna Maxi Dress",
        "product_type": "Dress",
        "product_description": "Flowing elegance meets modern sophistication. The Luna features a dramatic slit, cinched waist, and delicate shoulder straps.",
        "colors_available": "Navy, Burgundy",
        "material": "Chiffon",
        "price": 529,
        "vibe_tags": ["gala ready", "timeless elegance", "red carpet moment", "sophisticated", "romantic dinner", "luxurious"],
        "mood_summary": "For moments that deserve to be remembered forever."
    },
    {
        "product_id": "PROD004",
        "product_name": "Stella Jumpsuit",
        "product_type": "Jumpsuit",
        "product_description": "Bold and contemporary. The Stella jumpsuit features a plunging neckline, wide-leg silhouette, and statement belt.",
        "colors_available": "Black, Red",
        "material": "Crepe",
        "price": 479,
        "vibe_tags": ["power move", "boss babe energy", "modern elegance", "confident", "cocktail event", "statement piece"],
        "mood_summary": "For the woman who leads, not follows."
    },
    {
        "product_id": "PROD005",
        "product_name": "Aria Midi Dress",
        "product_type": "Dress",
        "product_description": "Effortlessly chic for day or night. The Aria features a flattering wrap design, flutter sleeves, and a midi length.",
        "colors_available": "Sage Green, Dusty Pink",
        "material": "Cotton Blend",
        "price": 299,
        "vibe_tags": ["effortlessly chic", "garden party", "brunch ready", "feminine", "day to night", "versatile"],
        "mood_summary": "Easy elegance for every occasion."
    },
    {
        "product_id": "PROD006",
        "product_name": "Eva Slip Dress",
        "product_type": "Dress",
        "product_description": "Understated luxury. The Eva features a cowl neckline, adjustable straps, and bias-cut silhouette that flatters every figure.",
        "colors_available": "Black, Champagne",
        "material": "Silk",
        "price": 379,
        "vibe_tags": ["quiet luxury", "date night", "sensual", "minimalist", "elegant", "timeless"],
        "mood_summary": "Less is more, but make it luxurious."
    },
    {
        "product_id": "PROD007",
        "product_name": "Bella Off-Shoulder Dress",
        "product_type": "Dress",
        "product_description": "Romantic and feminine. The Bella features a dramatic off-shoulder neckline, fitted bodice, and flowy skirt.",
        "colors_available": "White, Blush Pink",
        "material": "Lace",
        "price": 419,
        "vibe_tags": ["wedding guest", "romantic", "feminine", "garden party", "summer elegance", "dreamy"],
        "mood_summary": "For the romantic at heart."
    },
    {
        "product_id": "PROD008",
        "product_name": "Jade Cocktail Dress",
        "product_type": "Dress",
        "product_description": "A modern take on the classic cocktail dress. Features asymmetric hemline, one-shoulder design, and sculpted bodice.",
        "colors_available": "Emerald, Black",
        "material": "Jersey",
        "price": 349,
        "vibe_tags": ["cocktail hour", "modern classic", "sophisticated", "figure flattering", "night out", "chic"],
        "mood_summary": "Classic silhouette, modern attitude."
    },
]


# =============================================================================
# SIMPLE SEARCH (no vector DB needed for demo)
# =============================================================================
def search_products(query: str, products: List[Dict]) -> List[Dict]:
    """Simple keyword + vibe search for demo"""
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    scored_products = []
    
    for product in products:
        score = 0
        
        # Check product name
        if query_lower in product["product_name"].lower():
            score += 10
        
        # Check description
        desc_lower = product["product_description"].lower()
        for word in query_words:
            if word in desc_lower:
                score += 2
        
        # Check vibe tags (most important for this demo!)
        for vibe in product["vibe_tags"]:
            vibe_lower = vibe.lower()
            if query_lower in vibe_lower:
                score += 8
            for word in query_words:
                if word in vibe_lower:
                    score += 3
        
        # Check mood summary
        if query_lower in product.get("mood_summary", "").lower():
            score += 5
        
        # Check colors
        if query_lower in product["colors_available"].lower():
            score += 4
        
        # Check material
        if query_lower in product["material"].lower():
            score += 3
        
        if score > 0:
            scored_products.append((score, product))
    
    # Sort by score and return top results
    scored_products.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored_products[:5]]


# =============================================================================
# LLM CHAT (optional - uses Groq if available)
# =============================================================================
def get_llm_response(query: str, products: List[Dict]) -> str:
    """Get LLM response using Groq (free tier)"""
    api_key = get_api_key("GROQ_API_KEY")
    
    if not api_key:
        return None
    
    try:
        from groq import Groq
        
        client = Groq(api_key=api_key)
        
        # Build context from products
        context = "\n\n".join([
            f"**{p['product_name']}** (MYR {p['price']})\n"
            f"Type: {p['product_type']}\n"
            f"Colors: {p['colors_available']}\n"
            f"Vibes: {', '.join(p['vibe_tags'][:5])}\n"
            f"Description: {p['product_description'][:200]}"
            for p in products[:3]
        ])
        
        response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are a friendly fashion assistant for ByNoemie, a luxury women's boutique in Malaysia.
Help customers find the perfect outfit. Be warm, helpful, and enthusiastic about fashion.
Keep responses concise (2-3 sentences). Always mention specific products from the context."""
                },
                {
                    "role": "user",
                    "content": f"Customer query: {query}\n\nAvailable products:\n{context}\n\nRecommend the best option(s):"
                }
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"LLM error: {e}")
        return None


# =============================================================================
# UI COMPONENTS
# =============================================================================
def display_product_card(product: Dict):
    """Display a product card"""
    with st.container():
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Placeholder image
            st.image(
                f"https://via.placeholder.com/200x250/f0e6e6/333333?text={product['product_name'].replace(' ', '+')}",
                use_container_width=True
            )
        
        with col2:
            st.markdown(f"### {product['product_name']}")
            st.markdown(f"**MYR {product['price']}** | {product['product_type']}")
            st.markdown(f"*{product['mood_summary']}*")
            st.markdown(f"**Colors:** {product['colors_available']}")
            st.markdown(f"**Material:** {product['material']}")
            
            # Vibe tags as badges
            vibes_html = " ".join([
                f'<span style="background-color: #f8e1e7; color: #8b4557; padding: 2px 8px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 0.8em;">{vibe}</span>'
                for vibe in product['vibe_tags'][:5]
            ])
            st.markdown(f"**Vibes:** {vibes_html}", unsafe_allow_html=True)
        
        st.divider()


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1>👗 ByNoemie Fashion Assistant</h1>
        <p style="color: #666;">Find your perfect outfit with AI-powered recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 💡 Try asking:")
        st.markdown("""
        - *"Something for a romantic dinner"*
        - *"Night out dress that sparkles"*
        - *"Elegant but not boring"*
        - *"Main character energy"*
        - *"Wedding guest outfit"*
        - *"Boss babe power look"*
        """)
        
        st.divider()
        
        st.markdown("### ⚙️ Settings")
        use_llm = st.checkbox(
            "Use AI Assistant",
            value=bool(get_api_key("GROQ_API_KEY")),
            help="Enable for more conversational responses (requires Groq API key)"
        )
        
        st.divider()
        
        st.markdown("### 📊 Demo Info")
        st.markdown(f"**Products:** {len(SAMPLE_PRODUCTS)}")
        st.markdown("**Database:** In-memory (demo)")
        
        if get_api_key("GROQ_API_KEY"):
            st.success("✅ Groq API connected")
        else:
            st.warning("⚠️ Add GROQ_API_KEY for AI chat")
    
    # Main chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "products" in message:
                for product in message["products"]:
                    display_product_card(product)
    
    # Chat input
    if prompt := st.chat_input("What are you looking for today?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Search products
        with st.spinner("Finding perfect matches..."):
            results = search_products(prompt, SAMPLE_PRODUCTS)
        
        # Generate response
        with st.chat_message("assistant"):
            if results:
                # Try LLM response if enabled
                llm_response = None
                if use_llm:
                    llm_response = get_llm_response(prompt, results)
                
                if llm_response:
                    st.markdown(llm_response)
                else:
                    st.markdown(f"I found **{len(results)}** items that match your vibe! Here are my top picks:")
                
                # Display products
                for product in results:
                    display_product_card(product)
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": llm_response or f"Found {len(results)} matching items!",
                    "products": results
                })
            else:
                no_results_msg = "I couldn't find an exact match, but let me show you some popular options!"
                st.markdown(no_results_msg)
                
                # Show random recommendations
                import random
                random_picks = random.sample(SAMPLE_PRODUCTS, min(3, len(SAMPLE_PRODUCTS)))
                for product in random_picks:
                    display_product_card(product)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": no_results_msg,
                    "products": random_picks
                })
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888;'>"
        "Built with ❤️ using LangChain, LangGraph & Streamlit | "
        "<a href='https://github.com/bynoemie'>View on GitHub</a>"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
