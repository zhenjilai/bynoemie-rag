"""
ByNoemie Multi-Agent Chatbot Architecture

Agents:
1. RouterAgent - Routes queries to appropriate specialist agent
2. DeflectionAgent - Handles off-topic, greetings, thanks
3. InfoAgent - Product info, recommendations, stock, policy, tracking
4. ActionAgent - Order create/modify/cancel with validation
5. ConfirmationAgent - Handles ORDER/DELETE/CHANGE confirmations

All agents share conversation history and state.
"""

import json
import re
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class AgentType(Enum):
    DEFLECTION = "deflection"
    INFO = "info"
    ACTION = "action"
    CONFIRMATION = "confirmation"


@dataclass
class SharedState:
    """Shared state across all agents - contains FULL conversation history"""
    conversation_history: List[Dict] = field(default_factory=list)
    current_product: Optional[str] = None
    current_product_data: Optional[Dict] = None
    current_user_id: str = "USR-001"
    pending_action: Optional[Dict] = None  # {type: create/modify/cancel, data: {...}}
    last_shown_products: List[Dict] = field(default_factory=list)
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add message to conversation history"""
        msg = {"role": role, "content": content}
        if metadata:
            msg["metadata"] = metadata
        self.conversation_history.append(msg)
    
    def get_recent_history(self, n: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        return self.conversation_history[-n:]
    
    def get_full_history(self) -> List[Dict]:
        """Get full conversation history"""
        return self.conversation_history
    
    def get_history_text(self, n: int = 10) -> str:
        """Get recent history as concatenated text for context extraction"""
        recent = self.get_recent_history(n)
        return " ".join([msg.get('content', '') for msg in recent]).lower()
    
    def extract_context(self) -> Dict:
        """Extract useful context from conversation history"""
        history_text = self.get_history_text(10)
        
        context = {
            'mentioned_occasions': [],
            'mentioned_categories': [],
            'mentioned_colors': [],
            'mentioned_sizes': [],
        }
        
        # Extract occasions
        occasions = ['gala', 'wedding', 'dinner', 'party', 'date', 'formal', 'casual', 
                    'cocktail', 'brunch', 'beach', 'vacation', 'office', 'work', 'prom']
        for occ in occasions:
            if occ in history_text:
                context['mentioned_occasions'].append(occ)
        
        # Extract categories
        categories = ['dress', 'jumpsuit', 'heel', 'heels', 'shoe', 'shoes', 'bag', 'bags', 'top', 'set']
        for cat in categories:
            if cat in history_text:
                context['mentioned_categories'].append(cat)
        
        # Extract colors
        colors = ['black', 'white', 'beige', 'red', 'pink', 'gold', 'navy', 'cream', 'maroon', 'nude', 'champagne']
        for color in colors:
            if color in history_text:
                context['mentioned_colors'].append(color)
        
        # Extract sizes
        sizes = ['xs', 'small', 'medium', 'large', 'xl', 's size', 'm size', 'l size']
        for size in sizes:
            if size in history_text:
                context['mentioned_sizes'].append(size)
        
        return context
    
    def set_current_product(self, product: Dict):
        """Set current product context"""
        self.current_product = product.get('product_name')
        self.current_product_data = product
    
    def clear_pending_action(self):
        """Clear pending action after completion"""
        self.pending_action = None


@dataclass
class AgentResponse:
    """Standard response from any agent"""
    message: str
    products_to_show: List[Dict] = field(default_factory=list)
    action_completed: bool = False
    requires_confirmation: bool = False
    metadata: Dict = field(default_factory=dict)


# =============================================================================
# ROUTER AGENT - Decides which agent to use
# =============================================================================
class RouterAgent:
    """
    Analyzes user query and conversation context to route to appropriate agent.
    Uses LLM for intelligent routing with fallback to keyword matching.
    """
    
    def __init__(self, openai_client, product_names: List[str]):
        self.client = openai_client
        self.product_names = product_names
    
    def route(self, query: str, state: SharedState) -> Tuple[AgentType, Dict]:
        """
        Route query to appropriate agent.
        Returns: (AgentType, extracted_info)
        """
        q = query.strip()
        q_lower = q.lower()
        
        # 1. Check for EXACT confirmation keywords first (single word only!)
        if q.upper() in ["ORDER", "DELETE", "CHANGE"]:
            return AgentType.CONFIRMATION, {"confirm_type": q.upper()}
        
        # 2. ORDER ACTION KEYWORDS - Route cancel/remove/modify requests to ACTION
        # These are REQUESTS to perform an action, not confirmations
        action_phrases = [
            'cancel my order', 'cancel order', 'cancel the order',
            'remove my order', 'remove order', 'remove the order',
            'delete my order', 'delete order', 'delete the order',
            'modify my order', 'modify order', 'modify the order',
            'change my order', 'change order', 'change the order',
            'i want to cancel', 'i want to remove', 'i want to delete',
            'i want to modify', 'i want to change',
            'i want to order', 'i would like to order', 'can i order',
            'place an order', 'make an order', 'buy the', 'purchase the'
        ]
        if any(phrase in q_lower for phrase in action_phrases):
            # Extract product name if mentioned
            product_mentioned = None
            for pname in self.product_names:
                if pname.lower() in q_lower:
                    product_mentioned = pname
                    break
            return AgentType.ACTION, {"product_mentioned": product_mentioned, "reason": "Order action request"}
        
        # 3. Fashion keyword safety check - NEVER deflect fashion queries
        fashion_keywords = [
            'dress', 'dresses', 'jumpsuit', 'jumpsuits', 'heel', 'heels', 'shoe', 'shoes',
            'bag', 'bags', 'top', 'tops', 'set', 'sets', 'outfit', 'outfits',
            'wear', 'wearing', 'style', 'styling', 'fashion', 'clothes', 'clothing',
            'gala', 'wedding', 'dinner', 'party', 'date', 'occasion', 'formal', 'casual',
            'recommend', 'suggestion', 'show me', 'looking for', 'browse',
            'stock', 'available', 'size', 'color', 'colour', 'price', 'cost'
        ]
        is_fashion_query = any(kw in q_lower for kw in fashion_keywords)
        
        # Check if it's a product name
        is_product_query = any(pname.lower() in q_lower for pname in self.product_names)
        
        # 4. Use LLM for intelligent routing
        if self.client:
            result = self._llm_route(query, state)
            if result:
                agent_type, extracted = result
                # Safety: If LLM says DEFLECTION but query has fashion keywords, override to INFO
                if agent_type == AgentType.DEFLECTION and (is_fashion_query or is_product_query):
                    return AgentType.INFO, extracted
                # Safety: If LLM says CONFIRMATION but query is more than one word, it's likely ACTION
                if agent_type == AgentType.CONFIRMATION and len(q.split()) > 1:
                    return AgentType.ACTION, extracted
                return result
        
        # 5. Fallback to keyword-based routing
        return self._keyword_route(q_lower, state)
    
    def _llm_route(self, query: str, state: SharedState) -> Optional[Tuple[AgentType, Dict]]:
        """Use LLM for intelligent routing"""
        
        # Build context
        current_product = state.current_product or "None"
        pending_action = state.pending_action
        # Extract context from conversation history
        context = state.extract_context()
        
        pending_str = f"Pending: {pending_action['type']} for {pending_action.get('data', {}).get('product_name', 'unknown')}" if pending_action else "None"
        
        system_prompt = f"""You are a router for a fashion boutique chatbot.

═══════════════════════════════════════════════════════════
CONTEXT:
═══════════════════════════════════════════════════════════
- Current Product: {current_product}
- Pending Action: {pending_str}
- Recent Products: {', '.join([p.get('product_name', '') for p in state.last_shown_products[-3:]]) or 'None'}

═══════════════════════════════════════════════════════════
AGENTS - READ CAREFULLY:
═══════════════════════════════════════════════════════════
1. DEFLECTION - ONLY for truly off-topic (weather, math, cooking)

2. INFO - Fashion queries: product info, colors, sizes, stock, recommendations

3. ACTION - User REQUESTS to do something:
   • "i want to order..." → ACTION
   • "cancel my order" → ACTION
   • "remove my order" → ACTION  
   • "delete my order" → ACTION
   • "modify my order" → ACTION
   • "change my order" → ACTION
   
4. CONFIRMATION - ONLY when message is EXACTLY one word:
   • "ORDER" (exactly) → CONFIRMATION
   • "DELETE" (exactly) → CONFIRMATION
   • "CHANGE" (exactly) → CONFIRMATION
   • NOT "i want to order" (that's ACTION!)
   • NOT "cancel my order" (that's ACTION!)

═══════════════════════════════════════════════════════════
CRITICAL DISTINCTION:
═══════════════════════════════════════════════════════════
• "i want to cancel my order for tiara dress" → ACTION (user is REQUESTING cancellation)
• "DELETE" → CONFIRMATION (user is CONFIRMING a pending action)

• "remove my order" → ACTION
• "ORDER" → CONFIRMATION

═══════════════════════════════════════════════════════════
PRODUCTS: {', '.join(self.product_names[:15])}
═══════════════════════════════════════════════════════════

Return JSON:
{{"agent": "ACTION", "product_mentioned": "Product Name or null", "order_id": "ORD-XXX or null", "reason": "brief reason"}}"""""

        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add recent conversation for context
        for msg in state.get_recent_history(6):
            messages.append({"role": msg["role"], "content": msg["content"][:300]})
        
        messages.append({"role": "user", "content": f"Route this query: {query}"})
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=150,
                temperature=0.1
            )
            
            result = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                agent_str = parsed.get("agent", "INFO").upper()
                
                # Map to AgentType
                agent_map = {
                    "DEFLECTION": AgentType.DEFLECTION,
                    "INFO": AgentType.INFO,
                    "ACTION": AgentType.ACTION,
                    "CONFIRMATION": AgentType.CONFIRMATION
                }
                agent_type = agent_map.get(agent_str, AgentType.INFO)
                
                # Extract info
                extracted = {
                    "product_mentioned": parsed.get("product_mentioned"),
                    "size": parsed.get("size"),
                    "color": parsed.get("color"),
                    "order_id": parsed.get("order_id"),
                    "reason": parsed.get("reason")
                }
                
                # If no product mentioned but refers to current context
                if not extracted["product_mentioned"] and state.current_product:
                    q_lower = query.lower()
                    if any(w in q_lower for w in ['this', 'it', 'that', 'the dress', 'the item', 'for this', 'other color', 'other size']):
                        extracted["product_mentioned"] = state.current_product
                
                return agent_type, extracted
                
        except Exception as e:
            print(f"Router LLM error: {e}")
        
        return None
    
    def _keyword_route(self, q: str, state: SharedState) -> Tuple[AgentType, Dict]:
        """Fallback keyword-based routing"""
        
        extracted = {
            "product_mentioned": None,
            "size": None,
            "color": None,
            "order_id": None
        }
        
        # Extract product mention
        for name in self.product_names:
            if name.lower() in q:
                extracted["product_mentioned"] = name
                break
        
        # If no product but refers to current
        if not extracted["product_mentioned"] and state.current_product:
            if any(w in q for w in ['this', 'it', 'that', 'the dress', 'the item', 'for this']):
                extracted["product_mentioned"] = state.current_product
        
        # Extract size
        size_patterns = [
            (r'\bxs\b', 'XS'), (r'\bs\b(?!\w)', 'S'), (r'\bm\b(?!\w)', 'M'),
            (r'\bl\b(?!\w)', 'L'), (r'\bxl\b', 'XL'), (r'\b(36|37|38|39|40|41|42)\b', None)
        ]
        for pattern, size in size_patterns:
            match = re.search(pattern, q)
            if match:
                extracted["size"] = size or match.group(1)
                break
        
        # Extract color
        colors = ['black', 'white', 'beige', 'red', 'pink', 'gold', 'navy', 'cream', 'maroon', 'nude']
        for c in colors:
            if c in q:
                extracted["color"] = c.capitalize()
                break
        
        # Extract order ID
        order_match = re.search(r'ord-?\d{3}', q, re.IGNORECASE)
        if order_match:
            extracted["order_id"] = order_match.group().upper().replace('ORD', 'ORD-') if '-' not in order_match.group() else order_match.group().upper()
        
        # Determine agent
        # Greetings/Thanks/Goodbye
        if any(w in q for w in ['hello', 'hi', 'hey']) and len(q.split()) < 5:
            return AgentType.DEFLECTION, extracted
        if any(w in q for w in ['thank', 'thanks', 'bye', 'goodbye']):
            return AgentType.DEFLECTION, extracted
        
        # Off-topic check
        fashion_words = ['dress', 'jumpsuit', 'heel', 'bag', 'top', 'set', 'stock', 'order', 'buy',
                         'price', 'size', 'color', 'colour', 'ship', 'return', 'refund', 'track',
                         'recommend', 'show', 'style', 'fashion', 'wear', 'outfit', 'occasion']
        if not any(w in q for w in fashion_words) and not extracted["product_mentioned"]:
            return AgentType.DEFLECTION, extracted
        
        # Action intents
        action_keywords = ['want to order', 'i want', 'buy', 'purchase', 'place order', 
                          'cancel order', 'cancel my order', 'modify order', 'change order',
                          'update order', 'delete order', 'update profile', 'change my']
        if any(w in q for w in action_keywords):
            return AgentType.ACTION, extracted
        
        # Info intents (default for fashion queries)
        return AgentType.INFO, extracted


# =============================================================================
# DEFLECTION AGENT - Handles off-topic, greetings, thanks
# =============================================================================
class DeflectionAgent:
    """
    Handles:
    - Greetings (hi, hello)
    - Thanks
    - Goodbye
    - Off-topic questions (politely redirects)
    
    SAFETY: If fashion keywords detected, returns recommendation instead of deflecting!
    """
    
    def __init__(self, openai_client=None, products: List[Dict] = None):
        self.client = openai_client
        self.products = products or []
    
    def handle(self, query: str, state: SharedState, extracted: Dict) -> AgentResponse:
        q = query.lower().strip()
        
        # SAFETY CHECK: If this looks like a fashion query, show products instead of deflecting!
        fashion_keywords = ['dress', 'dresses', 'jumpsuit', 'heel', 'heels', 'bag', 'bags',
                           'wear', 'outfit', 'style', 'fashion', 'gala', 'wedding', 'dinner',
                           'party', 'formal', 'casual', 'cocktail', 'recommend', 'show', 'looking for']
        if any(kw in q for kw in fashion_keywords):
            # This is a fashion query - redirect to showing products!
            return self._show_products_fallback(query, state)
        
        # Greetings
        if any(w in q for w in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return AgentResponse(
                message="Hello! 👋 Welcome to ByNoemie! I'm here to help you find the perfect outfit. Are you looking for dresses, jumpsuits, heels, or something else today?",
                metadata={"intent": "greeting"}
            )
        
        # Thanks
        if any(w in q for w in ['thank', 'thanks']):
            return AgentResponse(
                message="You're welcome! 💕 Is there anything else I can help you with? Feel free to ask about our dresses, heels, or any other items!",
                metadata={"intent": "thanks"}
            )
        
        # Goodbye
        if any(w in q for w in ['bye', 'goodbye', 'see you', 'take care']):
            return AgentResponse(
                message="Goodbye! 👋 Thank you for visiting ByNoemie. Come back soon for more fabulous fashion! 💕",
                metadata={"intent": "goodbye"}
            )
        
        # Off-topic - simple redirect, no LLM needed
        return AgentResponse(
            message="I'm ByNoemie's fashion assistant! 👗 I can help you find dresses, jumpsuits, heels, and bags. What are you looking for today?",
            metadata={"intent": "off_topic"}
        )
    
    def _show_products_fallback(self, query: str, state: SharedState) -> AgentResponse:
        """Fallback: show products when fashion query incorrectly routed here"""
        if not self.products:
            return AgentResponse(
                message="I'd love to help you find something! What type of outfit are you looking for - dresses, jumpsuits, heels, or bags?"
            )
        
        # Determine category
        q = query.lower()
        category = 'Dress'  # Default
        if any(w in q for w in ['jumpsuit']):
            category = 'jumpsuit'
        elif any(w in q for w in ['heel', 'heels', 'shoe']):
            category = 'heel'
        elif any(w in q for w in ['bag', 'bags']):
            category = 'bag'
        
        # Filter and get products
        matching = [p for p in self.products if 
                   category.lower() in p.get('product_type', '').lower() 
                   or category.lower() in p.get('product_name', '').lower()]
        
        if not matching:
            matching = self.products
        
        matching = sorted(matching, key=lambda x: x.get('created_at', ''), reverse=True)[:10]
        
        # Update state
        state.last_shown_products = matching
        
        # Detect occasion
        occasion = ""
        for occ in ['gala', 'wedding', 'dinner', 'party', 'date', 'formal']:
            if occ in q:
                occasion = f" for your {occ}"
                break
        
        product_names = ", ".join([p['product_name'] for p in matching[:2]])
        return AgentResponse(
            message=f"Here are some stunning {category.lower()}s{occasion}! Check out the {product_names}. 💕",
            products_to_show=matching
        )


# =============================================================================
# INFO AGENT - Product info, recommendations, stock, policy, tracking
# =============================================================================
class InfoAgent:
    """
    Handles:
    - Product recommendations
    - Product info (colors, sizes, materials, price)
    - Stock availability
    - Policy questions (shipping, returns)
    - Order tracking/status
    """
    
    def __init__(self, openai_client, products: List[Dict], stock_data: Dict, 
                 order_manager=None, policy_rag=None):
        self.client = openai_client
        self.products = products
        self.stock_data = stock_data
        self.order_manager = order_manager
        self.policy_rag = policy_rag
        
        # Build product lookup
        self.product_lookup = {p['product_name'].lower(): p for p in products}
        
        # Debug: Print stock_data info
        print(f"🔍 InfoAgent initialized with {len(stock_data)} stock entries")
        if 'coco dress' in stock_data:
            print(f"   ✅ 'coco dress' found in stock_data")
        else:
            print(f"   ❌ 'coco dress' NOT in stock_data. Keys: {list(stock_data.keys())[:3]}")
    
    def handle(self, query: str, state: SharedState, extracted: Dict) -> AgentResponse:
        """
        IMPROVED ROUTING: Product-specific queries take PRIORITY over generic keywords.
        
        Priority Order:
        1. Order tracking/policy (no product needed)
        2. Product mentioned + attribute question → PRODUCT INFO
        3. Product mentioned alone → PRODUCT INFO  
        4. Generic recommendations (occasion, category, etc.)
        """
        q = query.lower()
        print(f"\n🔍 InfoAgent.handle()")
        print(f"   Query: '{q}'")
        print(f"   Extracted: {extracted}")
        
        # =====================================================================
        # STEP 1: DETECT PRODUCT MENTION - Find BEST match (most words)
        # =====================================================================
        mentioned_product = None
        best_match_score = 0
        
        # First check if router already extracted a product
        if extracted.get('product_mentioned'):
            mentioned_product = self._find_product(extracted['product_mentioned'])
            if mentioned_product:
                print(f"   ✓ Found from extracted: {mentioned_product['product_name']}")
                best_match_score = 100  # High score for exact extraction
        
        # Search query for product names - find BEST match
        if not mentioned_product or best_match_score < 100:
            candidates = []
            
            for p in self.products:
                pname = p['product_name'].lower()
                pname_words = [w for w in pname.split() if len(w) > 2]
                
                # Exact full name match - highest priority
                if pname in q:
                    candidates.append((p, len(pname_words) * 10, "exact"))
                    continue
                
                # Count matching words
                matches = sum(1 for word in pname_words if word in q)
                if matches >= 2:
                    # Score based on match ratio
                    score = (matches / len(pname_words)) * matches
                    candidates.append((p, score, f"{matches}/{len(pname_words)} words"))
            
            # Sort by score descending, pick best
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                best = candidates[0]
                if best[1] > best_match_score:
                    mentioned_product = best[0]
                    print(f"   ✓ Best match ({best[2]}): {best[0]['product_name']}")
        
        # Check for context references (this, it, that)
        if not mentioned_product and state.current_product:
            context_refs = ['this', 'it', 'that', 'the dress', 'the item', 'same one', 'this one']
            if any(ref in q for ref in context_refs):
                mentioned_product = self._find_product(state.current_product)
                if mentioned_product:
                    print(f"   ✓ Context reference to: {mentioned_product['product_name']}")
        
        # =====================================================================
        # STEP 2: NON-PRODUCT QUERIES (handle first if no product context needed)
        # =====================================================================
        
        # Order tracking
        if any(w in q for w in ['track', 'where is my order', 'order status', 'check order', 'my order']):
            print("   → _handle_order_tracking")
            return self._handle_order_tracking(query, state, extracted)
        
        # Policy questions
        if any(w in q for w in ['return', 'refund', 'exchange', 'shipping', 'delivery', 'policy']):
            print("   → _handle_policy")
            return self._handle_policy(query, state)
        
        # =====================================================================
        # STEP 3: PRODUCT-SPECIFIC QUERIES (HIGHEST PRIORITY!)
        # If user mentioned a specific product → route to product info/stock
        # =====================================================================
        
        attribute_keywords = ['color', 'colour', 'size', 'sizes', 'stock', 'available', 
                              'material', 'fabric', 'price', 'cost', 'how much', 
                              'do you have', 'in stock', 'tell me about']
        
        if mentioned_product:
            # Update state with the mentioned product
            state.set_current_product(mentioned_product)
            extracted['product_mentioned'] = mentioned_product['product_name']
            
            # Check if asking about specific attribute
            if any(kw in q for kw in attribute_keywords):
                print(f"   → _handle_product_info (product + attribute)")
                return self._handle_product_info(query, state, extracted)
            
            # Just mentioned product - show its info
            print(f"   → _handle_product_info (product mentioned)")
            return self._handle_product_info(query, state, extracted)
        
        # =====================================================================
        # STEP 4: GENERIC QUERIES (no specific product mentioned)
        # =====================================================================
        
        # What to wear questions
        if any(w in q for w in ['what to wear', 'what should i wear', 'outfit for', 
                                'something for', 'need something', 'looking for']):
            print("   → _handle_recommendation (what to wear)")
            return self._handle_recommendation(query, state, extracted)
        
        # Occasion-based recommendations
        occasion_words = ['gala', 'wedding', 'dinner', 'party', 'date', 'formal', 'casual', 
                         'cocktail', 'brunch', 'beach', 'vacation', 'office', 'work']
        if any(w in q for w in occasion_words):
            print("   → _handle_recommendation (occasion)")
            return self._handle_recommendation(query, state, extracted)
        
        # Category queries (short queries about dress types)
        category_words = ['dress', 'dresses', 'jumpsuit', 'jumpsuits', 'heel', 'heels', 
                         'shoe', 'shoes', 'bag', 'bags', 'top', 'tops', 'set', 'sets']
        words = q.split()
        if len(words) <= 5 and any(cw in words for cw in category_words):
            print("   → _handle_recommendation (category)")
            return self._handle_recommendation(query, state, extracted)
        
        # Explicit recommendation requests
        if any(w in q for w in ['recommend', 'suggest', 'show me', 'show', 'browse', 
                                'what do you have', 'any', 'some']):
            print("   → _handle_recommendation (explicit)")
            return self._handle_recommendation(query, state, extracted)
        
        # Attribute question without product - use context or ask
        if any(kw in q for kw in attribute_keywords):
            if state.current_product:
                extracted['product_mentioned'] = state.current_product
                print(f"   → _handle_product_info (attribute + context: {state.current_product})")
                return self._handle_product_info(query, state, extracted)
            return AgentResponse(
                message="Which product would you like to know about? Please mention the product name, or browse our collection! 💕"
            )
        
        # Default: recommendation
        print("   → _handle_recommendation (default)")
        return self._handle_recommendation(query, state, extracted)
    
    def _find_product(self, name: str) -> Optional[Dict]:
        """Find product by name (case insensitive, partial match)"""
        if not name:
            return None
        name_lower = name.lower()
        # Exact match
        if name_lower in self.product_lookup:
            return self.product_lookup[name_lower]
        # Partial match
        for pname, product in self.product_lookup.items():
            if name_lower in pname or pname in name_lower:
                return product
        return None
    
    def _handle_order_tracking(self, query: str, state: SharedState, extracted: Dict) -> AgentResponse:
        """Handle order tracking queries"""
        order_id = extracted.get('order_id')
        
        if not self.order_manager:
            return AgentResponse(message="Order tracking is currently unavailable. Please contact support.")
        
        if order_id:
            tracking_info = self.order_manager.track_order(order_id)
            return AgentResponse(message=tracking_info, metadata={"order_id": order_id})
        
        # No order ID - show recent orders
        user_orders = self.order_manager.get_orders_by_user(state.current_user_id)
        if user_orders:
            orders_list = "\n".join([
                f"• **{o['order_id']}**: {o['product_name']} - {o['status'].replace('_', ' ').title()}"
                for o in user_orders[:5]
            ])
            return AgentResponse(
                message=f"📦 **Your Recent Orders:**\n\n{orders_list}\n\nWhich order would you like to track? (e.g., 'Track ORD-001')"
            )
        
        return AgentResponse(message="I couldn't find any orders. Please provide your order ID (e.g., ORD-001).")
    
    def _handle_policy(self, query: str, state: SharedState) -> AgentResponse:
        """Handle policy questions using RAG if available"""
        if self.policy_rag:
            try:
                answer = self.policy_rag.query(query)
                return AgentResponse(message=answer)
            except:
                pass
        
        # Fallback policy responses
        q = query.lower()
        if 'return' in q or 'refund' in q:
            return AgentResponse(message="📋 **Return Policy**: Items can be returned within 14 days of delivery. Please contact us at support@bynoemie.com with your order number.")
        if 'shipping' in q or 'delivery' in q:
            return AgentResponse(message="🚚 **Shipping**: We ship within Malaysia. Standard delivery takes 3-7 business days. Express delivery (1-3 days) available for select areas.")
        
        return AgentResponse(message="For policy questions, please visit our website or contact support@bynoemie.com")
    
    def _handle_recommendation(self, query: str, state: SharedState, extracted: Dict) -> AgentResponse:
        """Handle product recommendations - ALWAYS shows products, uses FULL conversation history"""
        q = query.lower()
        
        # Use SharedState's context extraction (uses FULL history)
        context = state.extract_context()
        history_text = state.get_history_text(10)  # Last 10 messages as text
        
        # Determine category from query OR conversation context
        category = None
        
        # Check current query first
        if any(w in q for w in ['dress', 'dresses', 'gown']):
            category = 'Dress'
        elif any(w in q for w in ['jumpsuit', 'jumpsuits', 'romper']):
            category = 'jumpsuits'
        elif any(w in q for w in ['heel', 'heels', 'shoe', 'shoes']):
            category = 'Heel'
        elif any(w in q for w in ['bag', 'bags', 'clutch', 'purse']):
            category = 'bag'
        elif any(w in q for w in ['top', 'tops', 'blouse']):
            category = 'top'
        elif any(w in q for w in ['set', 'sets', 'coord']):
            category = 'set'
        
        # Check conversation history context if no category in current query
        if not category and context['mentioned_categories']:
            cat_map = {'dress': 'Dress', 'jumpsuit': 'jumpsuits', 'heel': 'Heel', 
                      'heels': 'Heel', 'shoe': 'Heel', 'shoes': 'Heel',
                      'bag': 'bag', 'bags': 'bag', 'top': 'top', 'set': 'set'}
            for cat in context['mentioned_categories']:
                if cat in cat_map:
                    category = cat_map[cat]
                    break
        
        # For occasion-based queries, default to dresses
        formal_occasions = ['gala', 'wedding', 'dinner', 'party', 'cocktail', 'formal', 'event', 'prom', 'ball']
        if not category and any(occ in q for occ in formal_occasions):
            category = 'Dress'
        # Check history context for occasions
        if not category and context['mentioned_occasions']:
            category = 'Dress'
        
        # Default to dresses for wear/outfit questions
        if not category and any(w in q for w in ['wear', 'outfit', 'something']):
            category = 'Dress'
        
        # If still no category, default to dresses
        if not category:
            category = 'Dress'
        
        # Filter products by category
        matching = [p for p in self.products if 
                   category.lower() in p.get('product_type', '').lower() 
                   or category.lower() in p.get('product_collection', '').lower()
                   or category.lower() in p.get('product_name', '').lower()
                   or category.lower() in p.get('subcategory', '').lower()]
        
        # Also filter by occasion if mentioned
        if context['mentioned_occasions']:
            occasion_matches = []
            for p in matching:
                p_occasions = p.get('occasions', [])
                p_vibes = p.get('vibe_tags', [])
                if isinstance(p_occasions, str):
                    p_occasions = [p_occasions]
                if isinstance(p_vibes, str):
                    p_vibes = [p_vibes]
                # Check if product matches any mentioned occasion
                for occ in context['mentioned_occasions']:
                    if occ in str(p_occasions).lower() or occ in str(p_vibes).lower():
                        occasion_matches.append(p)
                        break
            if occasion_matches:
                matching = occasion_matches
        
        # If no matches, show all products
        if not matching:
            matching = self.products
        
        # Add variety: shuffle then take top 10 (so different products each time)
        if len(matching) > 10:
            # Shuffle and pick 10 for variety
            shuffled = matching.copy()
            random.shuffle(shuffled)
            matching = shuffled[:10]
        else:
            matching = matching[:10]
        
        if matching:
            # Update state
            state.last_shown_products = matching
            if len(matching) == 1:
                state.set_current_product(matching[0])
            
            # Get occasion from context
            occasion = context['mentioned_occasions'][0] if context['mentioned_occasions'] else None
            if not occasion:
                for occ in formal_occasions:
                    if occ in q:
                        occasion = occ
                        break
            
            # Build response with LLM - include conversation history for context!
            if self.client:
                product_list = "\n".join([
                    f"- {p['product_name']}: MYR {p.get('price_min', 0)}, Colors: {p.get('colors_available', 'N/A')}"
                    for p in matching
                ])
                
                occasion_context = f" for a {occasion}" if occasion else ""
                
                # Build conversation history for LLM
                conv_history = []
                for msg in state.get_recent_history(6):
                    conv_history.append({"role": msg.get("role", "user"), "content": msg.get("content", "")[:200]})
                
                try:
                    messages = [
                        {"role": "system", "content": f"""You are a helpful fashion assistant for ByNoemie boutique.

CONVERSATION CONTEXT: The user has been discussing{occasion_context if occasion else ' fashion'}.
Previous topics mentioned: {', '.join(context['mentioned_occasions']) or 'none'}
Categories discussed: {', '.join(context['mentioned_categories']) or 'none'}

IMPORTANT RULES:
1. You MUST present the products below - DO NOT ask clarifying questions!
2. Reference the conversation context (e.g., "For your {occasion}..." if occasion mentioned)
3. Mention 1-2 specific products by name
4. Be concise (2-3 sentences)
5. DO NOT ask "what style?" or "what occasion?" - just present the products!"""},
                    ]
                    
                    # Add conversation history
                    messages.extend(conv_history)
                    
                    # Add current request with products
                    messages.append({
                        "role": "user", 
                        "content": f"Current query: {query}\n\nProducts to recommend:\n{product_list}"
                    })
                    
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        max_tokens=150,
                        temperature=0.7
                    )
                    return AgentResponse(
                        message=response.choices[0].message.content,
                        products_to_show=matching
                    )
                except Exception as e:
                    print(f"LLM error in recommendation: {e}")
            
            # Fallback without LLM
            product_names = ", ".join([p['product_name'] for p in matching[:3]])
            occasion_text = f" for your {occasion}" if occasion else ""
            return AgentResponse(
                message=f"Here are some beautiful {category.lower()}s{occasion_text}! Check out the {product_names}. 💕",
                products_to_show=matching
            )
        
        return AgentResponse(
            message=f"Let me show you our latest {category.lower()}s!",
            products_to_show=self.products[:10]
        )
    
    def _handle_stock_check(self, query: str, state: SharedState, extracted: Dict) -> AgentResponse:
        """Handle stock availability queries"""
        product_name = extracted.get('product_mentioned')
        product = self._find_product(product_name)
        
        # If no product specified, try to find from query
        if not product:
            q_lower = query.lower()
            for p in self.products:
                pname = p['product_name'].lower()
                if pname in q_lower:
                    product = p
                    break
                # Partial match
                pname_parts = pname.split()
                if any(part in q_lower for part in pname_parts if len(part) > 3):
                    product = p
                    break
        
        # If still no product, use current context
        if not product and state.current_product:
            product = self._find_product(state.current_product)
        
        if not product:
            return AgentResponse(
                message="Which product would you like me to check stock for? Please mention the product name."
            )
        
        # Update context
        state.set_current_product(product)
        
        # Get stock info
        size = extracted.get('size')
        color = extracted.get('color')
        
        stock_info = self._get_stock_info(product, size, color)
        
        if self.client:
            try:
                messages = [
                    {"role": "system", "content": f"""You are a helpful fashion assistant for ByNoemie.
                    
The user is asking about stock/availability for {product['product_name']}.

STOCK INFORMATION:
{stock_info}

Answer their question directly using the stock information above. 
Be friendly, concise, and include the specific colors and sizes available with quantities."""},
                    {"role": "user", "content": query}
                ]
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=200,
                    temperature=0.7
                )
                return AgentResponse(
                    message=response.choices[0].message.content,
                    products_to_show=[product]
                )
            except Exception as e:
                print(f"LLM error in _handle_stock_check: {e}")
        
        return AgentResponse(message=stock_info, products_to_show=[product])
    
    def _handle_price(self, query: str, state: SharedState, extracted: Dict) -> AgentResponse:
        """Handle price queries"""
        product_name = extracted.get('product_mentioned')
        product = self._find_product(product_name) or (self._find_product(state.current_product) if state.current_product else None)
        
        if product:
            state.set_current_product(product)
            price = product.get('price_min', 0)
            currency = product.get('price_currency', 'MYR')
            return AgentResponse(
                message=f"The **{product['product_name']}** is priced at **{currency} {price:.2f}**. Would you like to order it?",
                products_to_show=[product]
            )
        
        return AgentResponse(message="Which product's price would you like to know?")
    
    def _handle_product_info(self, query: str, state: SharedState, extracted: Dict) -> AgentResponse:
        """Handle general product info queries - uses FULL conversation history"""
        product_name = extracted.get('product_mentioned')
        product = self._find_product(product_name)
        
        # If no product, try to find it directly in the query
        if not product:
            q_lower = query.lower()
            for p in self.products:
                pname = p['product_name'].lower()
                if pname in q_lower:
                    product = p
                    break
                # Also check partial matches (e.g., "coco" for "Coco Dress")
                pname_parts = pname.split()
                if any(part in q_lower for part in pname_parts if len(part) > 3):
                    product = p
                    break
        
        # If no product but we have current context from state
        if not product and state.current_product:
            product = self._find_product(state.current_product)
        
        # Search conversation history for product context
        if not product:
            history_text = state.get_history_text(10)
            for p in self.products:
                if p['product_name'].lower() in history_text:
                    product = p
                    break
        
        if product:
            state.set_current_product(product)
            
            # Get detailed stock info if available
            product_name_lower = product['product_name'].lower()
            stock_info = self.stock_data.get(product_name_lower, {})
            
            # Debug logging
            print(f"🔍 Looking up stock for: '{product_name_lower}'")
            print(f"   Stock data keys: {list(self.stock_data.keys())[:5]}")
            print(f"   Found stock_info: {bool(stock_info)}")
            if stock_info:
                print(f"   Variants: {stock_info.get('variants', [])}")
            
            # Build variant details if available
            variant_details = ""
            if stock_info and 'variants' in stock_info:
                variants = stock_info['variants']
                variant_lines = []
                for v in variants:
                    color = v.get('color', 'N/A')
                    size = v.get('size', 'N/A')
                    qty = v.get('quantity', 0)
                    status = "✓ In Stock" if qty > 0 else "✗ Out of Stock"
                    variant_lines.append(f"  • {color} / {size}: {qty} available {status}")
                variant_details = "\n".join(variant_lines)
            
            # Build product info
            info = {
                "name": product['product_name'],
                "price": f"{product.get('price_currency', 'MYR')} {product.get('price_min', 0)}",
                "colors": product.get('colors_available', 'N/A'),
                "sizes": product.get('size_options', 'N/A'),
                "material": product.get('material', 'N/A'),
                "description": product.get('product_description', '')[:200],
                "variant_details": variant_details,
                "total_inventory": stock_info.get('total_inventory', product.get('total_inventory', 'N/A'))
            }
            
            # Get context from history
            context = state.extract_context()
            
            if self.client:
                try:
                    # Build a focused, clear system prompt
                    system_prompt = f"""You are a friendly fashion assistant for ByNoemie, a Malaysian fashion boutique.

═══════════════════════════════════════════════════════════
PRODUCT: {info['name']}
═══════════════════════════════════════════════════════════
💰 Price: {info['price']}
🎨 Colors: {info['colors']}
📏 Sizes: {info['sizes']}
🧵 Material: {info['material']}
📦 Total Stock: {info['total_inventory']} units

STOCK DETAILS:
{info['variant_details'] if info['variant_details'] else 'Standard stock available'}
═══════════════════════════════════════════════════════════

INSTRUCTIONS:
1. Answer the user's question DIRECTLY using ONLY the product data above
2. If asked about colors → List the exact colors with quantities
3. If asked about sizes → List the exact sizes available
4. If asked about stock → Show variant availability
5. Keep response SHORT (2-3 sentences max)
6. Be warm and helpful, use emojis sparingly
7. End with a soft call-to-action (e.g., "Would you like to order?")

DO NOT:
- Recommend other products unless asked
- Make up information not in the data
- Give long explanations"""

                    messages = [{"role": "system", "content": system_prompt}]
                    
                    # Add recent conversation for context
                    for msg in state.get_recent_history(3):
                        messages.append({
                            "role": msg.get("role", "user"), 
                            "content": msg.get("content", "")[:200]
                        })
                    
                    messages.append({"role": "user", "content": query})
                    
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        max_tokens=150,
                        temperature=0.6  # Slightly lower for more focused answers
                    )
                    return AgentResponse(
                        message=response.choices[0].message.content,
                        products_to_show=[product]
                    )
                except Exception as e:
                    print(f"LLM error in _handle_product_info: {e}")
            
            return AgentResponse(
                message=f"**{info['name']}** - {info['price']}\n\n🎨 **Colors**: {info['colors']}\n📏 **Sizes**: {info['sizes']}\n📦 **Total Stock**: {info['total_inventory']} units\n\n**Stock by Variant:**\n{info['variant_details'] if info['variant_details'] else 'Contact us for stock details'}",
                products_to_show=[product]
            )
        
        return AgentResponse(message="Which product would you like to know more about?")
    
    def _get_stock_info(self, product: Dict, size: str = None, color: str = None) -> str:
        """Get stock information for a product"""
        # Use product_name (lowercase) as key - matches how stock_data is keyed
        product_name_lower = product.get('product_name', '').lower()
        stock = self.stock_data.get(product_name_lower, {})
        
        print(f"🔍 _get_stock_info: Looking for '{product_name_lower}'")
        print(f"   Stock found: {bool(stock)}")
        
        if not stock:
            # Fallback: check with product_handle too
            product_handle = product.get('product_handle', '')
            stock = self.stock_data.get(product_handle, {})
            print(f"   Fallback to handle '{product_handle}': {bool(stock)}")
        
        if not stock:
            return f"{product['product_name']} - Stock information not available. Please contact us."
        
        # Build detailed stock info from variants
        info = f"**{product['product_name']}**\n\n"
        
        if 'variants' in stock:
            info += "📦 **Stock by Variant:**\n"
            for v in stock['variants']:
                color = v.get('color', 'N/A')
                vsize = v.get('size', 'N/A')
                qty = v.get('quantity', 0)
                status = "✅ In Stock" if qty > 0 else "❌ Out of Stock"
                info += f"  • {color} / {vsize}: {qty} available {status}\n"
            info += f"\n**Total Inventory:** {stock.get('total_inventory', 'N/A')} units"
        else:
            # Fallback to product data
            available_sizes = product.get('size_options', '').split(',')
            available_colors = product.get('colors_available', '').split(',')
            info += f"Available sizes: {', '.join([s.strip() for s in available_sizes])}\n"
            info += f"Available colors: {', '.join([c.strip() for c in available_colors])}\n"
        
        return info


# =============================================================================
# ACTION AGENT - Order create/modify/cancel with validation
# =============================================================================
class ActionAgent:
    """
    Handles actions that modify data:
    - Create order (with stock/color validation)
    - Modify order (with status + stock check)
    - Cancel order (with status check)
    - Update profile
    
    Always validates before action and requires confirmation.
    """
    
    def __init__(self, openai_client, products: List[Dict], stock_data: Dict,
                 order_manager=None, user_manager=None):
        self.client = openai_client
        self.products = products
        self.stock_data = stock_data
        self.order_manager = order_manager
        self.user_manager = user_manager
        
        self.product_lookup = {p['product_name'].lower(): p for p in products}
    
    def handle(self, query: str, state: SharedState, extracted: Dict) -> AgentResponse:
        q = query.lower()
        
        # Determine action type
        cancel_keywords = ['cancel order', 'cancel my order', 'delete order', 'delete my order',
                          'remove order', 'remove my order', 'remove the order']
        if any(w in q for w in cancel_keywords):
            return self._handle_cancel_order(query, state, extracted)
        
        modify_keywords = ['modify order', 'modify my order', 'change order', 'change my order',
                          'update order', 'update my order', 'edit order', 'edit my order']
        if any(w in q for w in modify_keywords):
            return self._handle_modify_order(query, state, extracted)
        
        if any(w in q for w in ['update profile', 'change my address', 'update my info']):
            return self._handle_update_profile(query, state, extracted)
        
        # Default: Create order
        return self._handle_create_order(query, state, extracted)
    
    def _find_product(self, name: str) -> Optional[Dict]:
        if not name:
            return None
        name_lower = name.lower()
        if name_lower in self.product_lookup:
            return self.product_lookup[name_lower]
        for pname, product in self.product_lookup.items():
            if name_lower in pname or pname in name_lower:
                return product
        return None
    
    def _handle_create_order(self, query: str, state: SharedState, extracted: Dict) -> AgentResponse:
        """
        Handle order creation with validation:
        1. Find product (from query, extracted, context, or history)
        2. Validate size exists
        3. Validate color exists
        4. Check stock (optional)
        5. Store pending action
        6. Ask for confirmation
        """
        # Find product from extracted info
        product_name = extracted.get('product_mentioned')
        product = self._find_product(product_name)
        
        # Try to find from query if not in extracted
        if not product:
            for p in self.products:
                if p['product_name'].lower() in query.lower():
                    product = p
                    break
        
        # Use current context if no product found
        if not product and state.current_product:
            product = self._find_product(state.current_product)
        
        # Search conversation history for products mentioned
        if not product:
            history_text = state.get_history_text(10)
            for p in self.products:
                if p['product_name'].lower() in history_text:
                    product = p
                    break
        
        # Check last shown products
        if not product and state.last_shown_products:
            product = state.last_shown_products[0]  # Use first shown product
        
        if not product:
            return AgentResponse(
                message="""📝 I'd love to help you place an order!

Please specify the product you'd like to order. For example:
• "I want to order the Luna Dress in size S, White"
• "Order the Ella Dress"

What would you like to order?"""
            )
        
        # Update context
        state.set_current_product(product)
        
        # Validate and get size
        available_sizes = [s.strip() for s in product.get('size_options', 'M').split(',')]
        size = extracted.get('size')
        
        # Extract size from query if not in extracted
        if not size:
            size_match = re.search(r'\b(xs|s|m|l|xl|36|37|38|39|40|41|42|free\s*size)\b', query.lower())
            if size_match:
                size = size_match.group(1).upper()
        
        if not size:
            size = available_sizes[0] if available_sizes else 'M'
        
        # Validate size
        size_valid = size.upper() in [s.upper() for s in available_sizes] or size in available_sizes
        if not size_valid:
            return AgentResponse(
                message=f"❌ Size **{size}** is not available for {product['product_name']}.\n\nAvailable sizes: **{', '.join(available_sizes)}**\n\nPlease choose a valid size.",
                products_to_show=[product]
            )
        
        # Validate and get color
        available_colors = [c.strip() for c in product.get('colors_available', 'Default').split(',')]
        color = extracted.get('color')
        
        if not color:
            color_match = re.search(r'\b(black|white|beige|red|pink|gold|navy|cream|maroon|nude|champagne|gray|grey)\b', query.lower())
            if color_match:
                color = color_match.group(1).capitalize()
        
        if not color:
            color = available_colors[0] if available_colors else 'Default'
        
        # Validate color
        color_valid = color.lower() in [c.lower() for c in available_colors]
        if not color_valid and color != 'Default':
            return AgentResponse(
                message=f"❌ Color **{color}** is not available for {product['product_name']}.\n\nAvailable colors: **{', '.join(available_colors)}**\n\nPlease choose a valid color.",
                products_to_show=[product]
            )
        
        # Extract quantity
        quantity = 1
        qty_match = re.search(r'(\d+)\s*(pcs|pieces|units)?', query.lower())
        if qty_match:
            qty = int(qty_match.group(1))
            if 1 <= qty <= 10:
                quantity = qty
        
        # Calculate price
        unit_price = product.get('price_min', 0)
        total_price = unit_price * quantity
        currency = product.get('price_currency', 'MYR')
        
        # Store pending action
        state.pending_action = {
            'type': 'create',
            'data': {
                'product': product,
                'product_name': product['product_name'],
                'product_id': product.get('product_id'),
                'size': size,
                'color': color,
                'quantity': quantity,
                'unit_price': unit_price,
                'total_price': total_price,
                'currency': currency
            }
        }
        
        return AgentResponse(
            message=f"""📝 **Order Summary**

• **Product:** {product['product_name']}
• **Product ID:** {product.get('product_id')}
• **Size:** {size} ✅
• **Color:** {color} ✅
• **Quantity:** {quantity}
• **Price:** {currency} {total_price:.2f}

⚠️ **To confirm your order, please type:** `ORDER`

_Type anything else to cancel._""",
            products_to_show=[product],
            requires_confirmation=True,
            metadata={"pending_action": "create_order"}
        )
    
    def _handle_modify_order(self, query: str, state: SharedState, extracted: Dict) -> AgentResponse:
        """
        Handle order modification:
        1. Find order
        2. Check status (must be pending/confirmed/processing)
        3. Validate new size/color if changing
        4. Store pending action
        5. Ask for confirmation
        """
        if not self.order_manager:
            return AgentResponse(message="Order management is currently unavailable.")
        
        order_id = extracted.get('order_id')
        
        if not order_id:
            # List modifiable orders
            user_orders = self.order_manager.get_orders_by_user(state.current_user_id)
            modifiable = [o for o in user_orders if self.order_manager.can_modify_order(o['order_id'])[0]]
            
            if modifiable:
                orders_list = "\n".join([f"• **{o['order_id']}**: {o['product_name']} - Size {o['size']}, {o['color']}" for o in modifiable[:5]])
                return AgentResponse(
                    message=f"📋 **Orders you can modify:**\n\n{orders_list}\n\nWhich order would you like to modify? (e.g., 'Change ORD-002 to size M')"
                )
            return AgentResponse(message="You don't have any orders that can be modified.")
        
        # Check if order can be modified
        can_modify, reason = self.order_manager.can_modify_order(order_id)
        if not can_modify:
            return AgentResponse(message=f"❌ Cannot modify order {order_id}. {reason}")
        
        order = self.order_manager.get_order(order_id)
        
        # Extract changes
        new_size = extracted.get('size')
        new_color = extracted.get('color')
        
        # Get product for validation
        product = self._find_product(order['product_name'])
        
        if new_size and product:
            available_sizes = [s.strip() for s in product.get('size_options', '').split(',')]
            if new_size.upper() not in [s.upper() for s in available_sizes]:
                return AgentResponse(message=f"❌ Size {new_size} not available. Available: {', '.join(available_sizes)}")
        
        if new_color and product:
            available_colors = [c.strip() for c in product.get('colors_available', '').split(',')]
            if new_color.lower() not in [c.lower() for c in available_colors]:
                return AgentResponse(message=f"❌ Color {new_color} not available. Available: {', '.join(available_colors)}")
        
        changes = {}
        changes_desc = []
        if new_size and new_size != order['size']:
            changes['size'] = new_size
            changes_desc.append(f"Size: {order['size']} → {new_size}")
        if new_color and new_color != order['color']:
            changes['color'] = new_color
            changes_desc.append(f"Color: {order['color']} → {new_color}")
        
        if not changes:
            return AgentResponse(
                message=f"📋 **Order {order_id}**: {order['product_name']}\nCurrent: Size {order['size']}, {order['color']}\n\nWhat would you like to change? (e.g., 'change to size M' or 'change to black')"
            )
        
        # Store pending action
        state.pending_action = {
            'type': 'modify',
            'order_id': order_id,
            'changes': changes
        }
        
        return AgentResponse(
            message=f"""✏️ **Modify Order {order_id}**

**Current:** {order['product_name']} - Size {order['size']}, {order['color']}

**Changes:**
{chr(10).join(['• ' + c for c in changes_desc])}

⚠️ **To confirm changes, please type:** `CHANGE`

_Type anything else to cancel._""",
            requires_confirmation=True,
            metadata={"pending_action": "modify_order"}
        )
    
    def _handle_cancel_order(self, query: str, state: SharedState, extracted: Dict) -> AgentResponse:
        """
        Handle order cancellation:
        1. Find order
        2. Check status (cannot cancel shipped/delivered)
        3. Store pending action
        4. Ask for confirmation
        """
        if not self.order_manager:
            return AgentResponse(message="Order management is currently unavailable.")
        
        order_id = extracted.get('order_id')
        product_mentioned = extracted.get('product_mentioned')
        
        # If no order_id but product mentioned, try to find order by product
        if not order_id and product_mentioned:
            user_orders = self.order_manager.get_orders_by_user(state.current_user_id)
            # Find orders for this product
            matching_orders = [
                o for o in user_orders 
                if product_mentioned.lower() in o.get('product_name', '').lower()
                and self.order_manager.can_cancel_order(o['order_id'])[0]
            ]
            if len(matching_orders) == 1:
                order_id = matching_orders[0]['order_id']
            elif len(matching_orders) > 1:
                orders_list = "\n".join([
                    f"• **{o['order_id']}**: {o['product_name']} ({o.get('size', 'N/A')}/{o.get('color', 'N/A')}) - {o['status'].replace('_', ' ').title()}" 
                    for o in matching_orders
                ])
                return AgentResponse(
                    message=f"📋 **Multiple orders found for {product_mentioned}:**\n\n{orders_list}\n\nWhich order would you like to cancel? Please specify the Order ID."
                )
        
        if not order_id:
            # List cancellable orders
            user_orders = self.order_manager.get_orders_by_user(state.current_user_id)
            cancellable = [o for o in user_orders if self.order_manager.can_cancel_order(o['order_id'])[0]]
            
            if cancellable:
                orders_list = "\n".join([f"• **{o['order_id']}**: {o['product_name']} - {o['status'].replace('_', ' ').title()}" for o in cancellable[:10]])
                return AgentResponse(
                    message=f"📋 **Orders you can cancel:**\n\n{orders_list}\n\nWhich order would you like to cancel? (e.g., 'Cancel ORD-004')"
                )
            return AgentResponse(message="You don't have any orders that can be cancelled.")
        
        # Check if order can be cancelled
        can_cancel, reason = self.order_manager.can_cancel_order(order_id)
        if not can_cancel:
            return AgentResponse(message=f"❌ Cannot cancel order {order_id}. {reason}")
        
        order = self.order_manager.get_order(order_id)
        
        # Store pending action
        state.pending_action = {
            'type': 'cancel',
            'order_id': order_id
        }
        
        return AgentResponse(
            message=f"""🗑️ **Cancel Order {order_id}?**

**Order Details:**
• Product: {order['product_name']}
• Size: {order['size']} | Color: {order['color']}
• Price: {order['currency']} {order['total_price']:.2f}
• Status: {order['status'].replace('_', ' ').title()}

⚠️ **To confirm cancellation, please type:** `DELETE`

_This action cannot be undone. Refund will be processed in 3-5 business days._""",
            requires_confirmation=True,
            metadata={"pending_action": "cancel_order"}
        )
    
    def _handle_update_profile(self, query: str, state: SharedState, extracted: Dict) -> AgentResponse:
        """Handle profile update requests"""
        return AgentResponse(
            message="To update your profile, please contact our support team at support@bynoemie.com with your request. We'll help you update your information securely."
        )


# =============================================================================
# CONFIRMATION AGENT - Handles ORDER/DELETE/CHANGE confirmations
# =============================================================================
class ConfirmationAgent:
    """
    Handles confirmation keywords: ORDER, DELETE, CHANGE
    Executes the pending action from state.
    """
    
    def __init__(self, order_manager=None, user_manager=None):
        self.order_manager = order_manager
        self.user_manager = user_manager
    
    def handle(self, query: str, state: SharedState, extracted: Dict) -> AgentResponse:
        confirm_type = extracted.get('confirm_type', query.strip().upper())
        pending = state.pending_action
        
        if not pending:
            return AgentResponse(
                message="I don't have a pending action to confirm. How can I help you?"
            )
        
        if confirm_type == "ORDER" and pending.get('type') == 'create':
            return self._confirm_create_order(state)
        
        if confirm_type == "DELETE" and pending.get('type') == 'cancel':
            return self._confirm_cancel_order(state)
        
        if confirm_type == "CHANGE" and pending.get('type') == 'modify':
            return self._confirm_modify_order(state)
        
        return AgentResponse(
            message=f"The confirmation '{confirm_type}' doesn't match the pending action ({pending.get('type')}). Please use the correct keyword."
        )
    
    def _confirm_create_order(self, state: SharedState) -> AgentResponse:
        """Execute order creation"""
        if not self.order_manager:
            state.clear_pending_action()
            return AgentResponse(message="Order system is temporarily unavailable. Please try again later.")
        
        data = state.pending_action.get('data', {})
        
        try:
            order = self.order_manager.create_order_simple(
                user_id=state.current_user_id,
                product=data.get('product', {}),
                size=data.get('size', 'M'),
                color=data.get('color', 'Default'),
                quantity=data.get('quantity', 1)
            )
            
            state.clear_pending_action()
            
            return AgentResponse(
                message=f"""✅ **Order Confirmed!**

Your order has been placed successfully!

**Order ID:** {order['order_id']}
**Product:** {order['product_name']}
**Size:** {order['size']} | **Color:** {order['color']}
**Total:** {order['currency']} {order['total_price']:.2f}

📧 You will receive a confirmation email shortly.
📦 Estimated delivery: {order['estimated_delivery']}

Thank you for shopping with ByNoemie! 💕""",
                action_completed=True,
                metadata={"order_id": order['order_id']}
            )
        except Exception as e:
            state.clear_pending_action()
            return AgentResponse(message=f"Error creating order: {str(e)}")
    
    def _confirm_cancel_order(self, state: SharedState) -> AgentResponse:
        """Execute order cancellation"""
        if not self.order_manager:
            state.clear_pending_action()
            return AgentResponse(message="Order system is temporarily unavailable.")
        
        order_id = state.pending_action.get('order_id')
        
        try:
            success, message, order = self.order_manager.cancel_order(order_id)
            state.clear_pending_action()
            
            if success:
                return AgentResponse(
                    message=f"""✅ **Order Cancelled**

Order **{order_id}** has been cancelled successfully.

💰 Your refund will be processed within 3-5 business days.

Is there anything else I can help you with?""",
                    action_completed=True
                )
            else:
                return AgentResponse(message=f"❌ {message}")
        except Exception as e:
            state.clear_pending_action()
            return AgentResponse(message=f"Error cancelling order: {str(e)}")
    
    def _confirm_modify_order(self, state: SharedState) -> AgentResponse:
        """Execute order modification"""
        if not self.order_manager:
            state.clear_pending_action()
            return AgentResponse(message="Order system is temporarily unavailable.")
        
        order_id = state.pending_action.get('order_id')
        changes = state.pending_action.get('changes', {})
        
        try:
            success, message, order = self.order_manager.modify_order(
                order_id,
                new_size=changes.get('size'),
                new_color=changes.get('color'),
                new_quantity=changes.get('quantity')
            )
            state.clear_pending_action()
            
            if success:
                return AgentResponse(
                    message=f"""✅ **Order Modified**

Order **{order_id}** has been updated successfully.

**Changes Applied:**
{message}

Is there anything else I can help you with?""",
                    action_completed=True
                )
            else:
                return AgentResponse(message=f"❌ {message}")
        except Exception as e:
            state.clear_pending_action()
            return AgentResponse(message=f"Error modifying order: {str(e)}")


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================
class ChatbotOrchestrator:
    """
    Main orchestrator that coordinates all agents.
    Maintains shared state and routes queries.
    """
    
    def __init__(self, openai_client, products: List[Dict], stock_data: Dict,
                 order_manager=None, user_manager=None, policy_rag=None):
        self.state = SharedState()
        
        # Extract product names
        product_names = [p['product_name'] for p in products]
        
        # Initialize agents
        self.router = RouterAgent(openai_client, product_names)
        self.deflection_agent = DeflectionAgent(openai_client, products)  # Pass products for fallback
        self.info_agent = InfoAgent(openai_client, products, stock_data, order_manager, policy_rag)
        self.action_agent = ActionAgent(openai_client, products, stock_data, order_manager, user_manager)
        self.confirmation_agent = ConfirmationAgent(order_manager, user_manager)
        
        self.agents = {
            AgentType.DEFLECTION: self.deflection_agent,
            AgentType.INFO: self.info_agent,
            AgentType.ACTION: self.action_agent,
            AgentType.CONFIRMATION: self.confirmation_agent
        }
    
    def process(self, query: str, chat_history: List[Dict] = None) -> AgentResponse:
        """
        Process a user query:
        1. Sync state from external history
        2. Route to appropriate agent
        3. Execute agent
        4. Update state
        5. Return response
        """
        print(f"\n{'='*50}")
        print(f"🤖 ORCHESTRATOR: Processing query: '{query}'")
        
        # Sync conversation history but PRESERVE pending_action
        saved_pending_action = self.state.pending_action
        
        if chat_history:
            self.state.conversation_history = []
            for msg in chat_history:
                self.state.add_message(
                    msg.get("role", "user"), 
                    msg.get("content", ""),
                    msg.get("metadata")
                )
        
        # Restore pending_action (it shouldn't be cleared by history sync)
        if saved_pending_action and not self.state.pending_action:
            self.state.pending_action = saved_pending_action
        
        # Add current user message to history
        self.state.add_message("user", query)
        
        # Route query
        agent_type, extracted = self.router.route(query, self.state)
        print(f"🎯 Router selected: {agent_type.value}")
        print(f"   Extracted: {extracted}")
        print(f"   Pending action: {self.state.pending_action}")
        
        # Get agent and process (ONLY ONCE!)
        agent = self.agents.get(agent_type, self.info_agent)
        print(f"📌 Calling agent: {type(agent).__name__}")
        response = agent.handle(query, self.state, extracted)
        print(f"💬 Response: {response.message[:100]}...")
        print(f"{'='*50}\n")
        
        # Update state
        self.state.add_message("assistant", response.message, {"agent": agent_type.value})
        
        # Update current product if products shown
        if response.products_to_show and len(response.products_to_show) == 1:
            self.state.set_current_product(response.products_to_show[0])
        
        if response.products_to_show:
            self.state.last_shown_products = response.products_to_show
        
        return response
    
    def set_user(self, user_id: str):
        """Set current user"""
        self.state.current_user_id = user_id
    
    def get_state(self) -> SharedState:
        """Get current state for inspection"""
        return self.state
    
    def clear_state(self):
        """Clear state for new conversation"""
        self.state = SharedState()
