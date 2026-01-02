"""
ByNoemie RAG Chatbot - FastAPI Backend
"""
import os
import json
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

# Import agents
from src.agents import ChatbotOrchestrator
from src.orders import OrderManager

# Initialize FastAPI
app = FastAPI(title="ByNoemie Fashion Assistant", version="2.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# DATA LOADING
# =============================================================================
def load_products():
    products_path = Path("data/products/bynoemie_products.json")
    if products_path.exists():
        with open(products_path, 'r') as f:
            return json.load(f)
    return []

def load_stock():
    """Load stock data - convert list to dict keyed by product_name"""
    stock_path = Path("data/stock/stock_inventory.json")
    if stock_path.exists():
        with open(stock_path, 'r') as f:
            stock_list = json.load(f)
            # Convert list to dict keyed by product_name (lowercase)
            if isinstance(stock_list, list):
                return {item['product_name'].lower(): item for item in stock_list}
            return stock_list
    return {}

def reload_stock():
    """Reload stock data from disk - call after order changes"""
    global stock_data
    stock_data = load_stock()
    # Also update agents' stock_data if orchestrator exists
    if orchestrator:
        orchestrator.info_agent.stock_data = stock_data
        orchestrator.action_agent.stock_data = stock_data
        print(f"🔄 Stock reloaded: {len(stock_data)} entries")
    return stock_data

def load_images():
    """Load image URLs - images are already in products data, build lookup by handle"""
    products_path = Path("data/products/bynoemie_products.json")
    if products_path.exists():
        with open(products_path, 'r') as f:
            products = json.load(f)
            # Build image lookup by product_handle
            images = {}
            for p in products:
                handle = p.get('product_handle', '')
                if handle:
                    images[handle] = {
                        'image_1': p.get('image_url_1', ''),
                        'image_2': p.get('image_url_2', ''),
                        'image_3': p.get('image_url_3', '')
                    }
            return images
    return {}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "bynoemie-chatbot"}

products = load_products()
stock_data = load_stock()
images_data = load_images()

# Debug: Print what we loaded
print(f"📦 Loaded {len(products)} products")
print(f"📊 Loaded {len(stock_data)} stock entries")
print(f"🖼️ Loaded {len(images_data)} image entries")

# Debug: Check if Coco Dress is in stock_data
if 'coco dress' in stock_data:
    print(f"✅ Coco Dress stock found: {stock_data['coco dress'].get('total_inventory')} units")
else:
    print(f"❌ Coco Dress NOT found in stock_data. Keys: {list(stock_data.keys())[:5]}")

# =============================================================================
# ORCHESTRATOR
# =============================================================================
orchestrator = None

def init_orchestrator():
    global orchestrator
    try:
        from openai import OpenAI
        
        # Try to load .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            print("⚠️ python-dotenv not installed, using system environment")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not found!")
            print("   Set it in .env file or as environment variable")
            return
        
        openai_client = OpenAI(api_key=api_key)
        order_manager = OrderManager()
        
        class SimpleUserManager:
            def get_user(self, user_id):
                return {"user_id": user_id, "name": "Customer"}
        
        class SimplePolicyRAG:
            def query(self, q):
                return "Standard return policy: 14 days for unworn items."
        
        orchestrator = ChatbotOrchestrator(
            openai_client=openai_client,
            products=products,
            stock_data=stock_data,
            order_manager=order_manager,
            user_manager=SimpleUserManager(),
            policy_rag=SimplePolicyRAG()
        )
        print("✅ Orchestrator initialized")
    except Exception as e:
        print(f"❌ Orchestrator error: {e}")

@app.on_event("startup")
async def startup():
    init_orchestrator()

# =============================================================================
# MODELS
# =============================================================================
class ChatRequest(BaseModel):
    message: str
    conversation_history: List[Dict] = []
    user_id: str = "USR-001"

class ChatResponse(BaseModel):
    message: str
    products: List[Dict] = []

# =============================================================================
# ENDPOINTS
# =============================================================================
@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = Path("static/index.html")
    if index_path.exists():
        return FileResponse("static/index.html")
    else:
        # Return inline HTML if file not found
        return HTMLResponse(content=get_inline_html(), status_code=200)

def get_inline_html():
    """Return the full HTML inline as fallback"""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ByNoemie Fashion Assistant</title>
    <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=Montserrat:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Montserrat', sans-serif; background: #0D0F12; color: #FFFFFF; min-height: 100vh; display: flex; flex-direction: column; }
        .header { text-align: center; padding: 30px 20px; background: linear-gradient(180deg, #1a1d24 0%, #0D0F12 100%); }
        .logo { font-family: 'Cormorant Garamond', serif; font-size: 2.5em; font-weight: 600; color: #D4A574; }
        .tagline { color: #888; font-size: 0.95em; margin-top: 5px; }
        .chat-container { flex: 1; max-width: 1200px; margin: 0 auto; width: 100%; padding: 20px; display: flex; flex-direction: column; }
        .messages { flex: 1; overflow-y: auto; padding-bottom: 20px; }
        .message { display: flex; margin-bottom: 20px; animation: fadeIn 0.3s ease; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .message.user { justify-content: flex-end; }
        .message-content { max-width: 80%; padding: 14px 18px; border-radius: 16px; line-height: 1.5; }
        .message.user .message-content { background: #D4A574; color: #0D0F12; border-bottom-right-radius: 4px; }
        .message.assistant .message-content { background: #1C1F26; color: #F5F5F5; border-bottom-left-radius: 4px; }
        .avatar { width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0; }
        .message.user .avatar { margin-left: 12px; background: #2A2F3A; order: 2; }
        .message.assistant .avatar { margin-right: 12px; background: #D4A574; }
        .product-carousel { display: flex; overflow-x: auto; gap: 16px; padding: 16px 4px; height: 380px; scroll-behavior: smooth; -webkit-overflow-scrolling: touch; scrollbar-width: none; }
        .product-carousel::-webkit-scrollbar { display: none; }
        .product-card { width: 200px; height: 340px; background: #1C1F26; border-radius: 14px; padding: 12px; display: flex; flex-direction: column; cursor: pointer; transition: all 0.3s ease; flex-shrink: 0; text-decoration: none; color: inherit; }
        .product-card:hover { transform: translateY(-4px); box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4); }
        .product-image-wrapper { width: 176px; height: 220px; overflow: hidden; border-radius: 10px; margin-bottom: 10px; background: #2A2F3A; }
        .product-image { width: 100%; height: 100%; object-fit: cover; transition: transform 0.3s ease; }
        .product-card:hover .product-image { transform: scale(1.03); }
        .product-name { font-size: 13px; font-weight: 600; color: #FFFFFF; margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .product-price { font-size: 12px; font-weight: 500; color: #B0F2C2; margin-bottom: 4px; }
        .product-category { font-size: 10px; color: #9AA0A6; margin-bottom: 6px; }
        .product-pills { display: flex; flex-wrap: wrap; gap: 4px; margin-top: auto; }
        .pill { display: inline-block; height: 18px; line-height: 18px; padding: 0 8px; border-radius: 9px; font-size: 10px; font-weight: 500; }
        .pill.stock-in { background: #1E3A2F; color: #4ADE80; }
        .pill.stock-low { background: #3A2A1E; color: #F2B04A; }
        .pill.stock-out { background: #3A2A2A; color: #888; }
        .pill.tag { background: #2A2F3A; color: #C7C7FF; }
        .input-area { padding: 20px; background: #1C1F26; border-top: 1px solid #2A2F3A; }
        .input-wrapper { max-width: 1200px; margin: 0 auto; display: flex; gap: 12px; }
        .chat-input { flex: 1; background: #0D0F12; border: 1px solid #2A2F3A; border-radius: 12px; padding: 14px 18px; color: #FFFFFF; font-family: 'Montserrat', sans-serif; font-size: 14px; outline: none; transition: border-color 0.3s ease; }
        .chat-input:focus { border-color: #D4A574; }
        .chat-input::placeholder { color: #666; }
        .send-btn { background: #D4A574; border: none; border-radius: 12px; padding: 14px 24px; color: #0D0F12; font-family: 'Montserrat', sans-serif; font-weight: 600; font-size: 14px; cursor: pointer; transition: all 0.3s ease; }
        .send-btn:hover { background: #E8C59D; transform: translateY(-2px); }
        .send-btn:disabled { background: #666; cursor: not-allowed; transform: none; }
        .quick-actions { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; justify-content: center; }
        .quick-btn { background: #1C1F26; border: 1px solid #2A2F3A; border-radius: 20px; padding: 8px 16px; color: #D4A574; font-size: 12px; cursor: pointer; transition: all 0.3s ease; }
        .quick-btn:hover { background: #2A2F3A; border-color: #D4A574; }
        .typing-indicator { display: flex; gap: 4px; padding: 10px; }
        .typing-dot { width: 8px; height: 8px; background: #D4A574; border-radius: 50%; animation: typing 1.4s infinite ease-in-out; }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typing { 0%, 60%, 100% { transform: translateY(0); } 30% { transform: translateY(-8px); } }
        .feedback-buttons { display: flex; gap: 8px; margin-top: 10px; }
        .feedback-btn { background: transparent; border: 1px solid #2A2F3A; border-radius: 8px; padding: 6px 12px; cursor: pointer; transition: all 0.2s ease; font-size: 16px; }
        .feedback-btn:hover { background: #2A2F3A; }
    </style>
</head>
<body>
    <header class="header">
        <h1 class="logo">✨ ByNoemie ✨</h1>
        <p class="tagline">Your Personal Fashion Assistant</p>
    </header>
    <div class="chat-container">
        <div class="quick-actions">
            <button class="quick-btn" onclick="sendMessage('What should I wear for a gala dinner?')">👗 Gala Dinner</button>
            <button class="quick-btn" onclick="sendMessage('Show me dresses')">💃 Dresses</button>
            <button class="quick-btn" onclick="sendMessage('Suggest outfit for date night')">💕 Date Night</button>
            <button class="quick-btn" onclick="sendMessage('What bags do you have?')">👜 Bags</button>
            <button class="quick-btn" onclick="sendMessage('Check my orders')">📦 My Orders</button>
        </div>
        <div class="messages" id="messages"></div>
    </div>
    <div class="input-area">
        <div class="input-wrapper">
            <input type="text" class="chat-input" id="chatInput" placeholder="How may I assist you?" onkeypress="handleKeyPress(event)">
            <button class="send-btn" id="sendBtn" onclick="sendMessage()">Send</button>
        </div>
    </div>
    <script>
        const messagesContainer = document.getElementById('messages');
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');
        let conversationHistory = [];

        function addMessage(content, role, products = []) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            const avatar = document.createElement('div');
            avatar.className = 'avatar';
            avatar.textContent = role === 'user' ? '👤' : '🛍️';
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.innerHTML = content;
            if (role === 'user') {
                messageDiv.appendChild(contentDiv);
                messageDiv.appendChild(avatar);
            } else {
                messageDiv.appendChild(avatar);
                messageDiv.appendChild(contentDiv);
                const feedbackDiv = document.createElement('div');
                feedbackDiv.className = 'feedback-buttons';
                feedbackDiv.innerHTML = '<button class="feedback-btn" onclick="sendFeedback(\'positive\')">👍</button><button class="feedback-btn" onclick="sendFeedback(\'negative\')">👎</button><button class="feedback-btn" onclick="sendFeedback(\'neutral\')">😐</button>';
                contentDiv.appendChild(feedbackDiv);
            }
            messagesContainer.appendChild(messageDiv);
            if (products && products.length > 0) {
                const carouselDiv = createProductCarousel(products);
                messagesContainer.appendChild(carouselDiv);
            }
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function createProductCarousel(products) {
            const carouselDiv = document.createElement('div');
            carouselDiv.className = 'product-carousel';
            products.forEach(product => {
                const card = document.createElement('a');
                card.className = 'product-card';
                card.href = product.product_url;
                card.target = '_blank';
                let stockPill = '';
                if (product.stock_status === 'In Stock') {
                    if (product.total_inventory > 0 && product.total_inventory <= 5) {
                        stockPill = `<span class="pill stock-low">Only ${product.total_inventory} left</span>`;
                    } else {
                        stockPill = '<span class="pill stock-in">In Stock</span>';
                    }
                } else {
                    stockPill = '<span class="pill stock-out">Out of Stock</span>';
                }
                const tagPills = (product.tags || []).map(tag => `<span class="pill tag">${tag}</span>`).join('');
                card.innerHTML = `
                    <div class="product-image-wrapper">
                        <img src="${product.image_url}" alt="${product.product_name}" class="product-image" onerror="this.style.display='none'">
                    </div>
                    <div class="product-name">${product.product_name}</div>
                    <div class="product-price">MYR ${product.price}</div>
                    ${product.category ? `<div class="product-category">${product.category}</div>` : ''}
                    <div class="product-pills">${stockPill}${tagPills}</div>
                `;
                carouselDiv.appendChild(card);
            });
            return carouselDiv;
        }

        function showTypingIndicator() {
            const typingDiv = document.createElement('div');
            typingDiv.className = 'message assistant';
            typingDiv.id = 'typing-indicator';
            typingDiv.innerHTML = '<div class="avatar">🛍️</div><div class="message-content"><div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div></div>';
            messagesContainer.appendChild(typingDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function hideTypingIndicator() {
            const indicator = document.getElementById('typing-indicator');
            if (indicator) indicator.remove();
        }

        async function sendMessage(customMessage = null) {
            const message = customMessage || chatInput.value.trim();
            if (!message) return;
            chatInput.value = '';
            addMessage(message, 'user');
            conversationHistory.push({ role: 'user', content: message });
            sendBtn.disabled = true;
            chatInput.disabled = true;
            showTypingIndicator();
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message, conversation_history: conversationHistory })
                });
                if (!response.ok) throw new Error('Network error');
                const data = await response.json();
                hideTypingIndicator();
                addMessage(data.message, 'assistant', data.products);
                conversationHistory.push({ role: 'assistant', content: data.message });
            } catch (error) {
                hideTypingIndicator();
                addMessage('Sorry, I encountered an error. Please try again.', 'assistant');
                console.error('Error:', error);
            } finally {
                sendBtn.disabled = false;
                chatInput.disabled = false;
                chatInput.focus();
            }
        }

        function handleKeyPress(event) { if (event.key === 'Enter') sendMessage(); }
        function sendFeedback(type) { console.log('Feedback:', type); }
        document.addEventListener('DOMContentLoaded', () => {
            addMessage("Hello! I'm your ByNoemie fashion assistant. How can I help you today? 👗✨", 'assistant');
        });
    </script>
</body>
</html>'''

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    
    try:
        orchestrator.set_user(request.user_id)
        
        response = orchestrator.process(
            request.message,
            chat_history=request.conversation_history
        )
        
        # Reload stock if an order action was completed (create/cancel/modify)
        if response.action_completed:
            reload_stock()
        
        formatted_products = []
        if response.products_to_show:
            for p in response.products_to_show:
                handle = p.get('product_handle', '')
                image_url = images_data.get(handle, {}).get('image_1', '') or p.get('image_url_1', '')
                tags = p.get('vibe_tags', []) or p.get('style_attributes', [])
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(',')]
                
                # Get updated stock info
                product_name_lower = p.get('product_name', '').lower()
                updated_stock = stock_data.get(product_name_lower, {})
                total_inv = updated_stock.get('total_inventory', p.get('total_inventory', 0))
                
                formatted_products.append({
                    "product_name": p.get('product_name', 'Product'),
                    "product_handle": handle,
                    "price": p.get('price_min', 0),
                    "stock_status": 'In Stock' if total_inv > 0 else 'Out of Stock',
                    "total_inventory": total_inv,
                    "category": p.get('subcategory', '') or p.get('product_type', ''),
                    "tags": tags[:2] if tags else [],
                    "image_url": image_url,
                    "product_url": p.get('product_link', f"https://bynoemie.com.my/products/{handle}")
                })
        
        return ChatResponse(message=response.message, products=formatted_products)
    
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health():
    return {"status": "healthy", "orchestrator": orchestrator is not None, "products": len(products)}

# Static files - create folder if needed
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Check if index.html exists
index_path = static_dir / "index.html"
if not index_path.exists():
    print("⚠️ static/index.html not found! Creating placeholder...")

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)
