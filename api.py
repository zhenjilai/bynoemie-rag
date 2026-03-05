"""
ByNoemie RAG Chatbot - FastAPI Backend
With Whisper Large-v3 STT + OpenAI TTS voice features
"""
import os
import io
import re
import json
import tempfile
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
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
# DATA LOADING (UNCHANGED)
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
            if isinstance(stock_list, list):
                return {item['product_name'].lower(): item for item in stock_list}
            return stock_list
    return {}

def reload_stock():
    """Reload stock data from disk - call after order changes"""
    global stock_data
    stock_data = load_stock()
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

print(f"📦 Loaded {len(products)} products")
print(f"📊 Loaded {len(stock_data)} stock entries")
print(f"🖼️ Loaded {len(images_data)} image entries")

if 'coco dress' in stock_data:
    print(f"✅ Coco Dress stock found: {stock_data['coco dress'].get('total_inventory')} units")
else:
    print(f"❌ Coco Dress NOT found in stock_data. Keys: {list(stock_data.keys())[:5]}")

# =============================================================================
# ORCHESTRATOR (UNCHANGED — shared OpenAI client reused for Whisper + TTS)
# =============================================================================
orchestrator = None
openai_client_global = None

def init_orchestrator():
    global orchestrator, openai_client_global
    try:
        from openai import OpenAI
        
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
        
        openai_client_global = OpenAI(api_key=api_key)
        order_manager = OrderManager()
        
        class SimpleUserManager:
            def get_user(self, user_id):
                return {"user_id": user_id, "name": "Customer"}
        
        class SimplePolicyRAG:
            def query(self, q):
                return "Standard return policy: 14 days for unworn items."
        
        orchestrator = ChatbotOrchestrator(
            openai_client=openai_client_global,
            products=products,
            stock_data=stock_data,
            order_manager=order_manager,
            user_manager=SimpleUserManager(),
            policy_rag=SimplePolicyRAG()
        )
        print("✅ Orchestrator initialized")
        print("🎤 Whisper Large-v3 STT ready (via OpenAI API)")
        print("🔊 OpenAI TTS-1 ready")
    except Exception as e:
        print(f"❌ Orchestrator error: {e}")

@app.on_event("startup")
async def startup():
    init_orchestrator()

# =============================================================================
# MODELS (UNCHANGED)
# =============================================================================
class ChatRequest(BaseModel):
    message: str
    conversation_history: List[Dict] = []
    user_id: str = "USR-001"

class ChatResponse(BaseModel):
    message: str
    products: List[Dict] = []

# =============================================================================
# VOICE ENDPOINTS (NEW — Whisper STT + OpenAI TTS)
# =============================================================================

@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe audio using OpenAI Whisper Large-v3.
    Accepts audio file (webm, wav, mp3, m4a, ogg, flac, mp4).
    Auto-detects language. Returns transcribed text.
    """
    if not openai_client_global:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    
    try:
        audio_bytes = await audio.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Determine filename extension
        filename = audio.filename or "audio.webm"
        valid_exts = ['.webm', '.wav', '.mp3', '.m4a', '.ogg', '.flac', '.mp4']
        if not any(filename.endswith(ext) for ext in valid_exts):
            filename = "audio.webm"
        
        # Create file-like object for OpenAI API
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename
        
        # Call Whisper via OpenAI API
        # "whisper-1" maps to Whisper Large-v3 on OpenAI's servers
        transcript = openai_client_global.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="json"
        )
        
        transcribed_text = transcript.text.strip()
        print(f"🎤 Whisper transcribed: '{transcribed_text}'")
        
        return {"text": transcribed_text, "status": "success"}
    
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/api/tts")
async def text_to_speech(
    text: str = Query(..., description="Text to convert to speech"),
    voice: str = Query(default="nova", description="Voice: alloy, echo, fable, onyx, nova, shimmer")
):
    """
    Convert text to speech using OpenAI TTS-1.
    Returns audio as MP3 stream.
    Supports any language — the model auto-detects.
    """
    if not openai_client_global:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    
    try:
        # Strip HTML tags
        clean_text = re.sub(r'<[^>]+>', '', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        if not clean_text:
            raise HTTPException(status_code=400, detail="Empty text")
        
        # TTS-1 has ~4096 char limit per request
        if len(clean_text) > 4096:
            clean_text = clean_text[:4096]
        
        # Validate voice
        valid_voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
        if voice not in valid_voices:
            voice = 'nova'
        
        # Call OpenAI TTS
        response = openai_client_global.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=clean_text,
            response_format="mp3"
        )
        
        audio_bytes = response.content
        print(f"🔊 TTS generated: {len(audio_bytes)} bytes, voice={voice}")
        
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=speech.mp3"}
        )
    
    except Exception as e:
        print(f"❌ TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


# =============================================================================
# ENDPOINTS (UNCHANGED)
# =============================================================================
@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = Path("static/index.html")
    if index_path.exists():
        return FileResponse("static/index.html")
    else:
        return HTMLResponse(content=get_inline_html(), status_code=200)

def get_inline_html():
    """Return full HTML inline as fallback — with Whisper STT + OpenAI TTS"""
    return '''
<!DOCTYPE html>
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
        .input-wrapper { max-width: 1200px; margin: 0 auto; display: flex; gap: 12px; align-items: center; }
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
        .mic-btn { background: #2A2F3A; border: 1px solid #3A3F4A; border-radius: 12px; padding: 14px 16px; color: #D4A574; font-size: 18px; cursor: pointer; transition: all 0.3s ease; display: flex; align-items: center; justify-content: center; }
        .mic-btn:hover { background: #3A3F4A; border-color: #D4A574; }
        .mic-btn.recording { background: #e74c3c; color: #FFF; border-color: #e74c3c; animation: pulse-mic 1.5s infinite; }
        .mic-btn.processing { background: #D4A574; color: #0D0F12; border-color: #D4A574; }
        @keyframes pulse-mic { 0%, 100% { box-shadow: 0 0 0 0 rgba(231,76,60,0.4); } 50% { box-shadow: 0 0 0 10px rgba(231,76,60,0); } }
        .voice-controls { display: flex; align-items: center; justify-content: center; gap: 12px; margin-top: 12px; }
        .voice-toggle-btn { display: flex; align-items: center; gap: 6px; background: #2A2F3A; border: 1px solid #D4A574; border-radius: 20px; padding: 6px 14px; color: #D4A574; font-family: 'Montserrat', sans-serif; font-size: 12px; cursor: pointer; transition: all 0.3s ease; }
        .voice-toggle-btn:hover { border-color: #D4A574; color: #D4A574; }
        .voice-toggle-btn.active { background: #2A2F3A; border-color: #D4A574; color: #D4A574; }
        .voice-select { background: #1C1F26; border: 1px solid #2A2F3A; border-radius: 20px; padding: 6px 12px; color: #888; font-family: 'Montserrat', sans-serif; font-size: 12px; cursor: pointer; outline: none; }
        .voice-select:focus { border-color: #D4A574; color: #D4A574; }
        .msg-speak-btn { background: transparent; border: 1px solid #2A2F3A; border-radius: 8px; padding: 6px 12px; cursor: pointer; transition: all 0.2s ease; font-size: 16px; }
        .msg-speak-btn:hover { background: #2A2F3A; }
        .msg-speak-btn.playing { color: #D4A574; border-color: #D4A574; }
        .voice-status { text-align: center; color: #D4A574; font-size: 12px; padding: 4px 0; min-height: 20px; }
        @media (max-width: 768px) { .logo { font-size: 2em; } .message-content { max-width: 90%; } .product-carousel { height: 340px; } .product-card { width: 160px; height: 300px; } .product-image-wrapper { width: 136px; height: 180px; } }
    </style>
</head>
<body>
    <header class="header">
        <h1 class="logo">✨ ByNoemie ✨</h1>
        <p class="tagline">Your Personal Fashion Assistant</p>
        <div class="voice-controls">
            <button class="voice-toggle-btn active" id="ttsToggle" onclick="toggleTTS()" title="Auto-read chatbot replies aloud">
                <span class="toggle-icon">🔊</span>
                <span id="ttsLabel">AI Voice: ON</span>
            </button>
            <select class="voice-select" id="voiceSelect" title="Select TTS voice">
                <option value="alloy">Alloy</option>
                <option value="nova" selected>Nova</option>
                <option value="shimmer">Shimmer</option>
                <option value="echo">Echo</option>
                <option value="fable">Fable</option>
                <option value="onyx">Onyx</option>
            </select>
        </div>
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
    <div class="voice-status" id="voiceStatus"></div>
    <div class="input-area">
        <div class="input-wrapper">
            <input type="text" class="chat-input" id="chatInput" placeholder="Type or press mic to speak..." onkeypress="handleKeyPress(event)">
            <button class="mic-btn" id="micBtn" onclick="toggleRecording()" title="Click to record — powered by Whisper AI">🎤</button>
            <button class="send-btn" id="sendBtn" onclick="sendMessage()">Send</button>
        </div>
    </div>
    <script>
        const messagesContainer = document.getElementById('messages');
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');
        let conversationHistory = [];

        // ============================================================
        // TTS pre-fetch cache: start fetching audio BEFORE message renders
        // This eliminates the latency the user would feel
        // ============================================================
        let ttsPrefetchCache = null; // { promise, text }

        function prefetchTTS(text) {
            const clean = cleanTextForTTSRaw(text);
            if (!clean) return;
            const params = new URLSearchParams({ text: clean, voice: voiceSelect.value });
            const promise = fetch('/api/tts?' + params, { method: 'POST' })
                .then(res => {
                    if (!res.ok) throw new Error('TTS prefetch failed');
                    return res.blob();
                })
                .then(blob => URL.createObjectURL(blob))
                .catch(err => { console.error('TTS prefetch error:', err); return null; });
            ttsPrefetchCache = { promise, text: clean };
        }

        // Clean text without needing a DOM element (for prefetch before render)
        function cleanTextForTTSRaw(text) {
            return text.replace(/<[^>]+>/g, '').replace(/\\s+/g, ' ').trim();
        }

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
                feedbackDiv.innerHTML = `<button class="feedback-btn" onclick="sendFeedback('positive')">👍</button><button class="feedback-btn" onclick="sendFeedback('negative')">👎</button><button class="feedback-btn" onclick="sendFeedback('neutral')">😐</button><button class="msg-speak-btn" onclick="speakMessage(this)" title="Read aloud (OpenAI TTS)">🔈</button>`;
                contentDiv.appendChild(feedbackDiv);

                // Auto-play TTS using prefetched audio (no extra latency)
                if (ttsEnabled && content) {
                    playPrefetchedOrFreshTTS(content);
                }
            }
            messagesContainer.appendChild(messageDiv);
            if (products && products.length > 0) {
                messagesContainer.appendChild(createProductCarousel(products));
            }
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function createProductCarousel(products) {
            const d = document.createElement('div');
            d.className = 'product-carousel';
            products.forEach(p => {
                const c = document.createElement('a');
                c.className = 'product-card'; c.href = p.product_url; c.target = '_blank';
                let sp = '';
                if (p.stock_status === 'In Stock') {
                    sp = (p.total_inventory > 0 && p.total_inventory <= 5) ? `<span class="pill stock-low">Only ${p.total_inventory} left</span>` : '<span class="pill stock-in">In Stock</span>';
                } else { sp = '<span class="pill stock-out">Out of Stock</span>'; }
                const tp = (p.tags||[]).map(t=>`<span class="pill tag">${t}</span>`).join('');
                c.innerHTML = `<div class="product-image-wrapper"><img src="${p.image_url}" alt="${p.product_name}" class="product-image" onerror="this.style.display='none'"></div><div class="product-name">${p.product_name}</div><div class="product-price">MYR ${p.price}</div>${p.category?`<div class="product-category">${p.category}</div>`:''}<div class="product-pills">${sp}${tp}</div>`;
                d.appendChild(c);
            });
            return d;
        }

        function showTypingIndicator() {
            const t = document.createElement('div');
            t.className = 'message assistant'; t.id = 'typing-indicator';
            t.innerHTML = '<div class="avatar">🛍️</div><div class="message-content"><div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div></div>';
            messagesContainer.appendChild(t);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        function hideTypingIndicator() { const i = document.getElementById('typing-indicator'); if (i) i.remove(); }

        async function sendMessage(customMessage = null) {
            const message = customMessage || chatInput.value.trim();
            if (!message) return;
            chatInput.value = '';
            addMessage(message, 'user');
            conversationHistory.push({ role: 'user', content: message });
            sendBtn.disabled = true; chatInput.disabled = true;
            showTypingIndicator();
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message, conversation_history: conversationHistory })
                });
                if (!response.ok) throw new Error('Network error');
                const data = await response.json();

                // === KEY LATENCY FIX ===
                // Start TTS fetch IMMEDIATELY when we get the text,
                // BEFORE rendering the message to DOM
                if (ttsEnabled && data.message) {
                    prefetchTTS(data.message);
                }

                hideTypingIndicator();
                addMessage(data.message, 'assistant', data.products);
                conversationHistory.push({ role: 'assistant', content: data.message });
            } catch (error) {
                hideTypingIndicator();
                addMessage('Sorry, I encountered an error. Please try again.', 'assistant');
                console.error('Error:', error);
            } finally {
                sendBtn.disabled = false; chatInput.disabled = false; chatInput.focus();
            }
        }

        function handleKeyPress(e) { if (e.key === 'Enter') sendMessage(); }
        function sendFeedback(type) { console.log('Feedback:', type); }

        /* ============================================================
           VOICE INPUT — Record audio → Whisper Large-v3 via /api/transcribe
           ============================================================ */
        let mediaRecorder = null, audioChunks = [], isRecording = false;
        const micBtn = document.getElementById('micBtn');
        const voiceStatus = document.getElementById('voiceStatus');

        async function toggleRecording() {
            if (isRecording) { stopRecording(); } else { await startRecording(); }
        }

        async function startRecording() {
            try {
                stopCurrentAudio();
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus'
                    : MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/mp4';
                mediaRecorder = new MediaRecorder(stream, { mimeType });
                audioChunks = [];
                mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
                mediaRecorder.onstop = async () => {
                    stream.getTracks().forEach(t => t.stop());
                    if (audioChunks.length === 0) { voiceStatus.textContent = 'No audio captured.'; setTimeout(()=>voiceStatus.textContent='',3000); return; }
                    micBtn.classList.remove('recording'); micBtn.classList.add('processing'); micBtn.textContent = '⏳';
                    voiceStatus.textContent = '🧠 Transcribing with Whisper AI...';
                    try {
                        const blob = new Blob(audioChunks, { type: mimeType });
                        const ext = mimeType.includes('webm') ? 'webm' : 'mp4';
                        const fd = new FormData(); fd.append('audio', blob, `recording.${ext}`);
                        const res = await fetch('/api/transcribe', { method: 'POST', body: fd });
                        if (!res.ok) { const err = await res.json().catch(()=>({})); throw new Error(err.detail || 'Transcription failed'); }
                        const data = await res.json();
                        if (data.text && data.text.trim()) { voiceStatus.textContent = ''; sendMessage(data.text.trim()); }
                        else { voiceStatus.textContent = 'Could not understand. Try again.'; setTimeout(()=>voiceStatus.textContent='',3000); }
                    } catch (err) { console.error(err); voiceStatus.textContent = 'Error: '+err.message; setTimeout(()=>voiceStatus.textContent='',4000); }
                    finally { micBtn.classList.remove('processing'); micBtn.textContent = '🎤'; isRecording = false; }
                };
                mediaRecorder.start(); isRecording = true;
                micBtn.classList.add('recording'); micBtn.textContent = '⏹️';
                voiceStatus.textContent = '🔴 Recording... Click again to stop';
            } catch (err) {
                console.error(err);
                voiceStatus.textContent = err.name === 'NotAllowedError' ? 'Mic access denied.' : 'Mic error: '+err.message;
                setTimeout(()=>voiceStatus.textContent='',4000);
            }
        }
        function stopRecording() { if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop(); }

        /* ============================================================
           VOICE OUTPUT — OpenAI TTS-1 via /api/tts
           ============================================================ */
        let ttsEnabled = true;  // <<< ON by default
        let currentAudio = null;
        const ttsToggle = document.getElementById('ttsToggle');
        const ttsLabel = document.getElementById('ttsLabel');
        const voiceSelect = document.getElementById('voiceSelect');

        function toggleTTS() {
            ttsEnabled = !ttsEnabled;
            if (ttsEnabled) {
                ttsToggle.classList.add('active');
                ttsLabel.textContent = 'AI Voice: ON';
                ttsToggle.querySelector('.toggle-icon').textContent = '🔊';
            } else {
                ttsToggle.classList.remove('active');
                ttsLabel.textContent = 'AI Voice: OFF';
                ttsToggle.querySelector('.toggle-icon').textContent = '🔇';
                stopCurrentAudio();
            }
        }

        function stopCurrentAudio() {
            if (currentAudio) { currentAudio.pause(); currentAudio.currentTime = 0; currentAudio = null; }
            document.querySelectorAll('.msg-speak-btn.playing').forEach(b => { b.classList.remove('playing'); b.textContent = '🔈'; });
        }

        function cleanTextForTTS(html) {
            const t = document.createElement('div'); t.innerHTML = html;
            t.querySelectorAll('.feedback-buttons').forEach(f=>f.remove());
            return (t.textContent||t.innerText||'').replace(/\\s+/g,' ').trim();
        }

        /**
         * Play TTS using prefetched audio if available, otherwise fetch fresh.
         * This is the key to eliminating latency — the audio is already
         * downloading while the message is being rendered.
         */
        async function playPrefetchedOrFreshTTS(html) {
            stopCurrentAudio();
            const text = cleanTextForTTSRaw(html);
            if (!text) return;

            try {
                let audioUrl = null;

                // Check if we have a prefetched audio that matches this text
                if (ttsPrefetchCache && ttsPrefetchCache.text === text) {
                    // Use the prefetched audio — already downloading or ready
                    audioUrl = await ttsPrefetchCache.promise;
                    ttsPrefetchCache = null;
                }

                // Fallback: fetch fresh if prefetch missed or failed
                if (!audioUrl) {
                    const params = new URLSearchParams({ text, voice: voiceSelect.value });
                    const res = await fetch('/api/tts?' + params, { method: 'POST' });
                    if (!res.ok) throw new Error('TTS failed');
                    const blob = await res.blob();
                    audioUrl = URL.createObjectURL(blob);
                }

                const audio = new Audio(audioUrl);
                currentAudio = audio;
                audio.onended = () => { currentAudio = null; URL.revokeObjectURL(audioUrl); };
                audio.onerror = () => { currentAudio = null; URL.revokeObjectURL(audioUrl); };
                await audio.play();
            } catch (err) {
                console.error('TTS playback error:', err);
            }
        }

        /**
         * Play TTS fresh (for per-message speaker button clicks)
         */
        async function playTTS(html, btnEl=null) {
            stopCurrentAudio();
            const text = cleanTextForTTS(html);
            if (!text) return;
            if (btnEl) { btnEl.classList.add('playing'); btnEl.textContent = '⏳'; }
            try {
                const params = new URLSearchParams({ text, voice: voiceSelect.value });
                const res = await fetch('/api/tts?'+params, { method: 'POST' });
                if (!res.ok) throw new Error('TTS failed');
                const blob = await res.blob();
                const url = URL.createObjectURL(blob);
                const audio = new Audio(url); currentAudio = audio;
                if (btnEl) btnEl.textContent = '🔊';
                audio.onended = () => { currentAudio=null; URL.revokeObjectURL(url); if(btnEl){btnEl.classList.remove('playing');btnEl.textContent='🔈';} };
                audio.onerror = () => { currentAudio=null; URL.revokeObjectURL(url); if(btnEl){btnEl.classList.remove('playing');btnEl.textContent='🔈';} };
                await audio.play();
            } catch (err) { console.error(err); if(btnEl){btnEl.classList.remove('playing');btnEl.textContent='🔈';} }
        }

        function speakMessage(btn) {
            const cd = btn.closest('.message-content'); if (!cd) return;
            if (btn.classList.contains('playing')) { stopCurrentAudio(); return; }
            playTTS(cd.innerHTML, btn);
        }

        document.addEventListener('DOMContentLoaded', () => {
            addMessage("Hello! I'm your ByNoemie fashion assistant. How can I help you today? 👗✨", 'assistant');
        });
    </script>
</body>
</html>
'''

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

# Static files
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

index_path = static_dir / "index.html"
if not index_path.exists():
    print("⚠️ static/index.html not found! Creating placeholder...")

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)