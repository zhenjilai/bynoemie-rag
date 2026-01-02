"""
ByNoemie Multi-Agent Chatbot Architecture - LLM-First Design

This version minimizes keyword-based routing and uses LLM for:
1. Intelligent routing with full context understanding
2. Intent and entity extraction
3. Sub-intent determination within agents
4. Response generation with conversation awareness

Agents:
1. RouterAgent - LLM-based routing with context awareness
2. DeflectionAgent - Handles off-topic, greetings, thanks
3. InfoAgent - Product info, recommendations, stock, policy, tracking
4. ActionAgent - Order create/modify/cancel with validation
5. ConfirmationAgent - Handles ORDER/DELETE/CHANGE confirmations
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
    pending_action: Optional[Dict] = None
    last_shown_products: List[Dict] = field(default_factory=list)
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        msg = {"role": role, "content": content}
        if metadata:
            msg["metadata"] = metadata
        self.conversation_history.append(msg)
    
    def get_recent_history(self, n: int = 10) -> List[Dict]:
        return self.conversation_history[-n:]
    
    def get_full_history(self) -> List[Dict]:
        return self.conversation_history
    
    def get_history_text(self, n: int = 10) -> str:
        recent = self.get_recent_history(n)
        return " ".join([msg.get('content', '') for msg in recent]).lower()
    
    def get_conversation_summary(self, n: int = 6) -> str:
        """Get formatted conversation history for LLM context"""
        recent = self.get_recent_history(n)
        summary = []
        for msg in recent:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")[:300]
            summary.append(f"{role}: {content}")
        return "\n".join(summary)
    
    def set_current_product(self, product: Dict):
        self.current_product = product.get('product_name')
        self.current_product_data = product
    
    def clear_pending_action(self):
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
# ROUTER AGENT - LLM-First Intelligent Routing
# =============================================================================
class RouterAgent:
    """
    LLM-first router that understands context, intent, and conversation flow.
    Only uses minimal keyword checks for single-word confirmations.
    """
    
    def __init__(self, openai_client, product_names: List[str]):
        self.client = openai_client
        self.product_names = product_names
    
    def route(self, query: str, state: SharedState) -> Tuple[AgentType, Dict]:
        """
        Route query using LLM for intelligent understanding.
        Only keyword check: single-word confirmations (ORDER, DELETE, CHANGE)
        """
        q = query.strip()
        
        # ONLY keyword check: exact single-word confirmations
        if q.upper() in ["ORDER", "DELETE", "CHANGE", "YES", "CONFIRM", "NO", "CANCEL"]:
            if q.upper() in ["NO", "CANCEL"]:
                state.clear_pending_action()
                return AgentType.DEFLECTION, {"intent": "cancel_action"}
            return AgentType.CONFIRMATION, {"confirm_type": q.upper()}
        
        # Everything else: LLM-based routing
        return self._llm_route(query, state)
    
    def _llm_route(self, query: str, state: SharedState) -> Tuple[AgentType, Dict]:
        """
        Use LLM for comprehensive intent understanding with full context.
        """
        # Build rich context
        conversation_history = state.get_conversation_summary(6)
        current_product = state.current_product or "None"
        
        pending_info = "None"
        if state.pending_action:
            pending_type = state.pending_action.get('type', 'unknown')
            pending_data = state.pending_action.get('data', {})
            pending_product = pending_data.get('product_name') or state.pending_action.get('product_name', '')
            pending_info = f"Awaiting confirmation for: {pending_type} - {pending_product}"
        
        last_products = ", ".join([p['product_name'] for p in state.last_shown_products[:3]]) if state.last_shown_products else "None"
        
        system_prompt = f"""You are an intelligent router for ByNoemie, a Malaysian fashion boutique chatbot.
Your job is to analyze the user's message IN CONTEXT of the conversation and determine:
1. Which agent should handle this request
2. Extract all relevant entities and intents

═══════════════════════════════════════════════════════════════════════════════
CURRENT STATE
═══════════════════════════════════════════════════════════════════════════════
• Current Product Context: {current_product}
• Recently Shown Products: {last_products}
• Pending Action: {pending_info}
• User ID: {state.current_user_id}

═══════════════════════════════════════════════════════════════════════════════
CONVERSATION HISTORY
═══════════════════════════════════════════════════════════════════════════════
{conversation_history}

═══════════════════════════════════════════════════════════════════════════════
CURRENT MESSAGE: "{query}"
═══════════════════════════════════════════════════════════════════════════════

AVAILABLE AGENTS:

1. **CONFIRMATION** - ONLY for single-word confirmations of pending actions
   - "ORDER", "DELETE", "CHANGE", "YES", "CONFIRM"
   - NOT for requests like "yes, show me more" or "order the blue one"

2. **ACTION** - User wants to PERFORM an order operation:
   - Create order: "I want to buy...", "order this", "purchase the..."
   - Modify order: "change my order", "switch to size M", "update ORD-123"
   - Cancel order: "cancel my order", "remove order", "delete ORD-456"
   - IMPORTANT: If user provides order IDs (ORD-XXXXX) after assistant asked "which order?", this is ACTION
   - IMPORTANT: "I want to order" = ACTION (create), NOT just INFO

3. **INFO** - User wants INFORMATION (no transaction):
   - Product details: "what colors?", "how much?", "tell me about..."
   - Stock queries: "is this available?", "how many in stock?"
   - Recommendations: "show me dresses", "what do you recommend?"
   - Order tracking: "where is my order?", "track order"
   - Policy questions: "return policy", "shipping info"

4. **DEFLECTION** - Off-topic or social:
   - Greetings: "hi", "hello"
   - Thanks: "thank you"
   - Goodbye: "bye"
   - Completely off-topic: weather, math, unrelated questions

═══════════════════════════════════════════════════════════════════════════════
CRITICAL CONTEXT RULES
═══════════════════════════════════════════════════════════════════════════════

1. **Follow-up Detection**: If the assistant just asked a question (e.g., "Which order would you like to cancel?") 
   and the user responds with relevant info (e.g., "ORD-39048"), route based on the ORIGINAL intent.

2. **Implicit References**: "this one", "it", "that dress" refer to {current_product or 'the last discussed product'}

3. **Action vs Info**: 
   - "I want to order the Luna Dress" → ACTION (create)
   - "Tell me about the Luna Dress" → INFO
   - "Is the Luna Dress available in black?" → INFO (stock query)
   - "Order the Luna Dress in black" → ACTION (create)

4. **Multiple Order IDs**: If user provides multiple order IDs like "ORD-123 and ORD-456", extract ALL of them.

═══════════════════════════════════════════════════════════════════════════════
PRODUCTS (for reference): {', '.join(self.product_names[:20])}
═══════════════════════════════════════════════════════════════════════════════

Return a JSON object with your analysis:
{{
    "agent": "ACTION|INFO|CONFIRMATION|DEFLECTION",
    "intent": "specific intent (e.g., create_order, check_stock, recommend, cancel_order, modify_order, greeting, track_order)",
    "action_subtype": "create|modify|cancel|null (only for ACTION agent)",
    "product_mentioned": "exact product name or null",
    "order_ids": ["ORD-XXXXX"] or null,
    "size": "XS|S|M|L|XL or null",
    "color": "color name or null",
    "quantity": number or null,
    "occasion": "occasion type or null",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of your routing decision"
}}"""

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": query})
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=300,
                temperature=0.1
            )
            
            result = response.choices[0].message.content
            print(f"🧠 Router LLM: {result}")
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                
                agent_str = parsed.get("agent", "INFO").upper()
                agent_map = {
                    "DEFLECTION": AgentType.DEFLECTION,
                    "INFO": AgentType.INFO,
                    "ACTION": AgentType.ACTION,
                    "CONFIRMATION": AgentType.CONFIRMATION
                }
                agent_type = agent_map.get(agent_str, AgentType.INFO)
                
                # Build extracted info
                extracted = {
                    "intent": parsed.get("intent"),
                    "action_subtype": parsed.get("action_subtype"),
                    "product_mentioned": parsed.get("product_mentioned"),
                    "order_id": self._normalize_order_ids(parsed.get("order_ids")),
                    "size": parsed.get("size"),
                    "color": parsed.get("color"),
                    "quantity": parsed.get("quantity"),
                    "occasion": parsed.get("occasion"),
                    "confidence": parsed.get("confidence", 0.5),
                    "reasoning": parsed.get("reasoning")
                }
                
                print(f"🎯 Routed to: {agent_type.value} | Intent: {extracted.get('intent')} | Confidence: {extracted.get('confidence')}")
                return agent_type, extracted
                
        except Exception as e:
            print(f"❌ Router LLM error: {e}")
        
        # Fallback: minimal keyword detection
        return self._fallback_route(query, state)
    
    def _normalize_order_ids(self, order_ids) -> Optional[str]:
        """Normalize order IDs to comma-separated string"""
        if not order_ids:
            return None
        if isinstance(order_ids, str):
            order_ids = [order_ids]
        
        normalized = []
        for oid in order_ids:
            oid = str(oid).upper().strip()
            if not oid.startswith("ORD-"):
                oid = "ORD-" + oid.replace("ORD", "")
            normalized.append(oid)
        
        return ",".join(normalized) if normalized else None
    
    def _fallback_route(self, query: str, state: SharedState) -> Tuple[AgentType, Dict]:
        """Minimal fallback when LLM fails - still tries to be smart"""
        q = query.lower()
        extracted = {"intent": "unknown", "fallback": True}
        
        # Check for order IDs
        order_ids = re.findall(r'ord-?\d{3,5}', q, re.IGNORECASE)
        if order_ids:
            extracted["order_id"] = ",".join([oid.upper() for oid in order_ids])
            return AgentType.ACTION, extracted
        
        # Simple intent detection
        if any(w in q for w in ['cancel', 'remove', 'delete', 'modify', 'change', 'order', 'buy', 'purchase']):
            return AgentType.ACTION, extracted
        
        if any(w in q for w in ['hello', 'hi', 'thanks', 'bye']):
            return AgentType.DEFLECTION, extracted
        
        return AgentType.INFO, extracted


# =============================================================================
# DEFLECTION AGENT - Handles off-topic, greetings, thanks
# =============================================================================
class DeflectionAgent:
    """Handles greetings, thanks, goodbye, and off-topic queries using LLM"""
    
    def __init__(self, openai_client=None, products: List[Dict] = None):
        self.client = openai_client
        self.products = products or []
    
    def handle(self, query: str, state: SharedState, extracted: Dict) -> AgentResponse:
        intent = extracted.get("intent", "")
        
        # Handle cancel action intent
        if intent == "cancel_action":
            state.clear_pending_action()
            return AgentResponse(
                message="No problem! I've cancelled that. How else can I help you today? 💕"
            )
        
        # Use LLM for natural response
        if self.client:
            return self._llm_response(query, state, extracted)
        
        # Fallback responses
        return AgentResponse(
            message="Hello! 👋 I'm ByNoemie's fashion assistant. I can help you find dresses, jumpsuits, heels, and bags. What are you looking for today?"
        )
    
    def _llm_response(self, query: str, state: SharedState, extracted: Dict) -> AgentResponse:
        """Generate natural response using LLM"""
        intent = extracted.get("intent", "greeting")
        
        system_prompt = f"""You are a friendly fashion assistant for ByNoemie, a Malaysian fashion boutique.

The user's intent is: {intent}

Respond appropriately:
- For greetings: Welcome them warmly, introduce yourself as ByNoemie's fashion assistant
- For thanks: Express gratitude, ask if there's anything else
- For goodbye: Wish them well, thank them for visiting
- For off-topic: Politely redirect to fashion topics

Keep responses SHORT (1-2 sentences), warm, and include relevant emojis.
Always end with an invitation to explore fashion if appropriate."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=100,
                temperature=0.7
            )
            return AgentResponse(message=response.choices[0].message.content)
        except Exception as e:
            print(f"DeflectionAgent LLM error: {e}")
            return AgentResponse(
                message="Hello! 👋 I'm here to help you find beautiful fashion at ByNoemie. What can I show you today?"
            )


# =============================================================================
# INFO AGENT - LLM-First Product Info, Recommendations, Stock, Policy
# =============================================================================
class InfoAgent:
    """
    Handles information queries using LLM for:
    - Understanding the specific information need
    - Generating contextual, helpful responses
    - Product recommendations with personality
    """
    
    def __init__(self, openai_client, products: List[Dict], stock_data: Dict, 
                 order_manager=None, policy_rag=None):
        self.client = openai_client
        self.products = products
        self.stock_data = stock_data
        self.order_manager = order_manager
        self.policy_rag = policy_rag
        self.product_lookup = {p['product_name'].lower(): p for p in products}
        print(f"📦 InfoAgent initialized with {len(products)} products, {len(stock_data)} stock entries")
    
    def handle(self, query: str, state: SharedState, extracted: Dict) -> AgentResponse:
        """
        LLM-first handling - determine sub-intent and respond appropriately
        """
        intent = extracted.get("intent", "")
        print(f"\n📋 InfoAgent.handle() | Intent: {intent}")
        
        # Find product if mentioned
        product = self._find_product(extracted.get("product_mentioned"))
        if not product and state.current_product:
            product = self._find_product(state.current_product)
        
        # Route based on intent from router
        if intent == "track_order":
            return self._handle_order_tracking(query, state, extracted)
        
        if intent in ["return_policy", "shipping_info", "policy"]:
            return self._handle_policy(query, state)
        
        if intent in ["check_stock", "availability"]:
            return self._handle_stock(query, state, extracted, product)
        
        if intent in ["product_info", "product_details"]:
            return self._handle_product_info(query, state, extracted, product)
        
        if intent in ["recommend", "browse", "show_products"]:
            return self._handle_recommendation(query, state, extracted)
        
        # Default: Use LLM to determine best response
        return self._llm_determine_response(query, state, extracted, product)
    
    def _find_product(self, name: str) -> Optional[Dict]:
        """Find product by name with fuzzy matching"""
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
            # Word-based matching
            name_words = set(name_lower.split())
            pname_words = set(pname.split())
            if len(name_words & pname_words) >= 2:
                return product
        
        return None
    
    def _llm_determine_response(self, query: str, state: SharedState, extracted: Dict, product: Optional[Dict]) -> AgentResponse:
        """Use LLM to determine the best response when intent is unclear"""
        
        # Build context
        product_context = ""
        if product:
            stock_info = self._get_stock_info(product)
            product_context = f"""
PRODUCT IN CONTEXT: {product['product_name']}
- Price: {product.get('price_currency', 'MYR')} {product.get('price_min', 0)}
- Colors: {product.get('colors_available', 'N/A')}
- Sizes: {product.get('size_options', 'N/A')}
- Stock: {stock_info}
"""
        
        recent_products = ""
        if state.last_shown_products:
            recent_products = "\nRECENTLY SHOWN: " + ", ".join([p['product_name'] for p in state.last_shown_products[:5]])
        
        system_prompt = f"""You are an expert fashion assistant for ByNoemie, a Malaysian fashion boutique.

CONVERSATION HISTORY:
{state.get_conversation_summary(4)}

{product_context}
{recent_products}

USER QUERY: "{query}"

Analyze what the user wants and provide a helpful, specific response.
- If asking about a product, provide relevant details
- If asking about availability/stock, check the stock info provided
- If browsing/exploring, suggest relevant products
- Be conversational, warm, and helpful
- Keep response concise (2-4 sentences)
- Use emojis sparingly but appropriately

Respond directly to the user."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            products_to_show = [product] if product else state.last_shown_products[:5]
            return AgentResponse(
                message=response.choices[0].message.content,
                products_to_show=products_to_show
            )
        except Exception as e:
            print(f"InfoAgent LLM error: {e}")
            return self._handle_recommendation(query, state, extracted)
    
    def _handle_order_tracking(self, query: str, state: SharedState, extracted: Dict) -> AgentResponse:
        """Handle order tracking with LLM response generation"""
        if not self.order_manager:
            return AgentResponse(message="Order tracking is currently unavailable. Please contact support@bynoemie.com")
        
        order_id = extracted.get("order_id")
        if order_id:
            # Handle single order ID
            oid = order_id.split(",")[0] if "," in order_id else order_id
            order = self.order_manager.get_order(oid)
            if order:
                return AgentResponse(
                    message=f"""📦 **Order {oid} Status**

• Product: {order['product_name']}
• Size: {order.get('size', 'N/A')} | Color: {order.get('color', 'N/A')}
• Status: **{order['status'].replace('_', ' ').title()}**
• Estimated Delivery: {order.get('estimated_delivery', 'Contact support')}

Is there anything else you'd like to know about your order?"""
                )
        
        # Show all user orders
        user_orders = self.order_manager.get_orders_by_user(state.current_user_id)
        if user_orders:
            orders_list = "\n".join([
                f"• **{o['order_id']}**: {o['product_name']} - {o['status'].replace('_', ' ').title()}"
                for o in user_orders[:5]
            ])
            return AgentResponse(
                message=f"📦 **Your Orders:**\n\n{orders_list}\n\nWould you like details on any specific order?"
            )
        
        return AgentResponse(message="I couldn't find any orders for your account. Need help placing an order?")
    
    def _handle_policy(self, query: str, state: SharedState) -> AgentResponse:
        """Handle policy questions with LLM"""
        # Try RAG first
        if self.policy_rag:
            try:
                answer = self.policy_rag.query(query)
                return AgentResponse(message=answer)
            except:
                pass
        
        # Use LLM with policy knowledge
        policy_info = """
BYNOEMIE POLICIES:
- Returns: 14-day return policy for unworn items with tags
- Exchanges: Available within 14 days, subject to stock
- Shipping: 3-7 business days within Malaysia, Express 1-3 days for select areas
- International: Contact support for international shipping
- Refunds: Processed within 5-7 business days after return received
"""
        
        system_prompt = f"""You are ByNoemie's customer service assistant.

{policy_info}

Answer the customer's policy question based on the information above.
Be clear, helpful, and concise."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=150,
                temperature=0.5
            )
            return AgentResponse(message=response.choices[0].message.content)
        except:
            return AgentResponse(message="For detailed policy information, please visit our website or contact support@bynoemie.com")
    
    def _handle_stock(self, query: str, state: SharedState, extracted: Dict, product: Optional[Dict]) -> AgentResponse:
        """Handle stock queries with detailed information"""
        if not product:
            # Try to find from query
            for p in self.products:
                if p['product_name'].lower() in query.lower():
                    product = p
                    break
        
        if not product:
            return AgentResponse(
                message="Which product would you like me to check stock for? Please mention the product name. 💕"
            )
        
        state.set_current_product(product)
        stock_info = self._get_stock_info(product)
        
        # Use LLM to generate natural response
        system_prompt = f"""You are ByNoemie's fashion assistant answering a stock availability question.

PRODUCT: {product['product_name']}
PRICE: {product.get('price_currency', 'MYR')} {product.get('price_min', 0)}

STOCK INFORMATION:
{stock_info}

Answer the customer's question about availability naturally.
- Mention specific sizes/colors that are available
- If something is out of stock, suggest alternatives
- Keep it friendly and helpful
- If good stock available, encourage ordering"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=200,
                temperature=0.7
            )
            return AgentResponse(
                message=response.choices[0].message.content,
                products_to_show=[product]
            )
        except Exception as e:
            # Fallback
            return AgentResponse(
                message=f"📦 **{product['product_name']} Stock:**\n\n{stock_info}",
                products_to_show=[product]
            )
    
    def _handle_product_info(self, query: str, state: SharedState, extracted: Dict, product: Optional[Dict]) -> AgentResponse:
        """Handle product information queries"""
        if not product:
            for p in self.products:
                if p['product_name'].lower() in query.lower():
                    product = p
                    break
        
        if not product and state.current_product:
            product = self._find_product(state.current_product)
        
        if not product:
            return AgentResponse(message="Which product would you like to know about?")
        
        state.set_current_product(product)
        stock_info = self._get_stock_info(product)
        
        # Build comprehensive product info
        product_data = f"""
PRODUCT: {product['product_name']}
PRICE: {product.get('price_currency', 'MYR')} {product.get('price_min', 0)}
COLORS: {product.get('colors_available', 'N/A')}
SIZES: {product.get('size_options', 'N/A')}
MATERIAL: {product.get('material', 'N/A')}
DESCRIPTION: {product.get('product_description', 'A beautiful piece from ByNoemie')[:200]}
STOCK: {stock_info}
"""
        
        system_prompt = f"""You are ByNoemie's fashion expert assistant.

{product_data}

USER QUESTION: {query}

Answer their specific question using the product information above.
- Be specific and helpful
- If they're asking generally, highlight key features
- Mention availability if relevant
- Keep it conversational (2-3 sentences)
- End with a soft call-to-action if appropriate"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=200,
                temperature=0.7
            )
            return AgentResponse(
                message=response.choices[0].message.content,
                products_to_show=[product]
            )
        except:
            return AgentResponse(
                message=f"✨ **{product['product_name']}** - {product.get('price_currency', 'MYR')} {product.get('price_min', 0)}\n\nColors: {product.get('colors_available', 'N/A')}\nSizes: {product.get('size_options', 'N/A')}",
                products_to_show=[product]
            )
    
    def _handle_recommendation(self, query: str, state: SharedState, extracted: Dict) -> AgentResponse:
        """Handle product recommendations with LLM personalization"""
        occasion = extracted.get("occasion")
        color = extracted.get("color")
        
        # Filter products based on extracted preferences
        matching = self.products.copy()
        
        # Apply filters if specified
        if occasion:
            matching = [p for p in matching if occasion.lower() in str(p.get('occasions', '')).lower() 
                       or occasion.lower() in str(p.get('vibe_tags', '')).lower()] or matching
        
        if color:
            matching = [p for p in matching if color.lower() in p.get('colors_available', '').lower()] or matching
        
        # Randomize for variety
        random.shuffle(matching)
        matching = matching[:10]
        
        state.last_shown_products = matching
        
        # Build product list for LLM
        product_list = "\n".join([
            f"- {p['product_name']}: MYR {p.get('price_min', 0)}, Colors: {p.get('colors_available', 'N/A')}"
            for p in matching[:5]
        ])
        
        system_prompt = f"""You are ByNoemie's fashion stylist assistant.

CONVERSATION CONTEXT:
{state.get_conversation_summary(3)}

AVAILABLE PRODUCTS TO RECOMMEND:
{product_list}

USER REQUEST: {query}

Create a personalized recommendation response:
- Reference any mentioned preferences (occasion, style, etc.)
- Mention 2-3 specific products by name
- Be enthusiastic but not over-the-top
- Keep it to 2-3 sentences
- Use 1-2 relevant emojis"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=150,
                temperature=0.8
            )
            return AgentResponse(
                message=response.choices[0].message.content,
                products_to_show=matching
            )
        except:
            product_names = ", ".join([p['product_name'] for p in matching[:3]])
            return AgentResponse(
                message=f"Here are some beautiful pieces for you! Check out the {product_names}. 💕",
                products_to_show=matching
            )
    
    def _get_stock_info(self, product: Dict) -> str:
        """Get formatted stock information for a product"""
        product_key = product['product_name'].lower()
        stock_data = self.stock_data.get(product_key, {})
        
        if not stock_data or 'variants' not in stock_data:
            return "Stock information not available - please contact us for availability"
        
        variants = stock_data['variants']
        total = sum(v.get('quantity', 0) for v in variants)
        
        lines = [f"Total Available: {total} units"]
        for v in variants:
            qty = v.get('quantity', 0)
            status = "✅" if qty > 0 else "❌"
            lines.append(f"  {v.get('color', 'N/A')} / {v.get('size', 'N/A')}: {qty} {status}")
        
        return "\n".join(lines)


# =============================================================================
# ACTION AGENT - LLM-Enhanced Order Operations
# =============================================================================
class ActionAgent:
    """
    Handles order operations with LLM for:
    - Understanding user intent within action context
    - Extracting order details
    - Generating clear confirmation prompts
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
        """Route to appropriate action handler based on extracted intent"""
        action_subtype = extracted.get("action_subtype")
        intent = extracted.get("intent", "")
        
        print(f"\n⚡ ActionAgent.handle() | Subtype: {action_subtype} | Intent: {intent}")
        
        # Route based on action subtype
        if action_subtype == "cancel" or intent == "cancel_order":
            return self._handle_cancel_order(query, state, extracted)
        
        if action_subtype == "modify" or intent == "modify_order":
            return self._handle_modify_order(query, state, extracted)
        
        # Default: create order
        return self._handle_create_order(query, state, extracted)
    
    def _find_product(self, name: str) -> Optional[Dict]:
        """Find product by name"""
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
        """Handle order creation with LLM assistance"""
        # Find product
        product = self._find_product(extracted.get("product_mentioned"))
        
        if not product and state.current_product:
            product = self._find_product(state.current_product)
        
        if not product and state.last_shown_products:
            product = state.last_shown_products[0]
        
        if not product:
            # Use LLM to ask for product
            return AgentResponse(
                message="I'd love to help you place an order! 🛍️ Which product would you like to order? You can mention the product name, or I can show you our collection."
            )
        
        state.set_current_product(product)
        
        # Get size and color
        available_sizes = [s.strip() for s in product.get('size_options', 'M').split(',')]
        available_colors = [c.strip() for c in product.get('colors_available', 'Default').split(',')]
        
        size = extracted.get("size")
        color = extracted.get("color")
        quantity = extracted.get("quantity") or 1
        
        # Validate or default
        if size and size.upper() not in [s.upper() for s in available_sizes]:
            return AgentResponse(
                message=f"❌ Size **{size}** isn't available for {product['product_name']}.\n\nAvailable sizes: **{', '.join(available_sizes)}**\n\nWhich size would you like?",
                products_to_show=[product]
            )
        
        if color and color.lower() not in [c.lower() for c in available_colors]:
            return AgentResponse(
                message=f"❌ **{color}** isn't available for {product['product_name']}.\n\nAvailable colors: **{', '.join(available_colors)}**\n\nWhich color would you prefer?",
                products_to_show=[product]
            )
        
        # Use defaults if not specified
        size = size or available_sizes[0]
        color = color or available_colors[0]
        
        # Check stock
        product_key = product['product_name'].lower()
        stock_info = self.stock_data.get(product_key, {})
        stock_available = True
        stock_qty = 0
        
        if stock_info and 'variants' in stock_info:
            for v in stock_info['variants']:
                if v.get('size', '').upper() == size.upper() and v.get('color', '').lower() == color.lower():
                    stock_qty = v.get('quantity', 0)
                    stock_available = stock_qty >= quantity
                    break
        
        if not stock_available:
            return AgentResponse(
                message=f"😔 Sorry, **{product['product_name']}** in {color}/{size} is currently out of stock.\n\nWould you like to try a different size or color?",
                products_to_show=[product]
            )
        
        # Calculate price
        price = product.get('price_min', 0)
        total = price * quantity
        currency = product.get('price_currency', 'MYR')
        
        # Store pending action
        state.pending_action = {
            'type': 'create',
            'data': {
                'product': product,
                'product_name': product['product_name'],
                'product_id': product.get('product_id'),
                'size': size.upper(),
                'color': color.capitalize(),
                'quantity': quantity,
                'unit_price': price,
                'total_price': total,
                'currency': currency
            }
        }
        
        return AgentResponse(
            message=f"""📝 **Order Summary**

• **Product:** {product['product_name']}
• **Size:** {size.upper()}
• **Color:** {color.capitalize()}
• **Quantity:** {quantity}
• **Total:** {currency} {total:.2f}

✅ Stock Available: {stock_qty} units

━━━━━━━━━━━━━━━━━━━━━━
**Type `ORDER` to confirm your purchase**
━━━━━━━━━━━━━━━━━━━━━━""",
            products_to_show=[product],
            requires_confirmation=True,
            metadata={"pending_action": "create_order"}
        )
    
    def _handle_modify_order(self, query: str, state: SharedState, extracted: Dict) -> AgentResponse:
        """Handle order modification"""
        if not self.order_manager:
            return AgentResponse(message="Order management is currently unavailable.")
        
        order_id = extracted.get("order_id")
        
        # Parse order ID if comma-separated (take first for modify)
        if order_id and "," in order_id:
            order_id = order_id.split(",")[0].strip()
        
        # If no order ID, show modifiable orders
        if not order_id:
            user_orders = self.order_manager.get_orders_by_user(state.current_user_id)
            modifiable = [o for o in user_orders if self.order_manager.can_modify_order(o['order_id'])[0]]
            
            if modifiable:
                orders_list = "\n".join([
                    f"• **{o['order_id']}**: {o['product_name']} - Size {o.get('size', 'N/A')}, {o.get('color', 'N/A')}"
                    for o in modifiable[:10]
                ])
                return AgentResponse(
                    message=f"📋 **Orders you can modify:**\n\n{orders_list}\n\nPlease tell me which order and what you'd like to change.\n\n*Example: \"Change ORD-12345 to size M\"*"
                )
            return AgentResponse(message="You don't have any orders that can be modified at this time.")
        
        # Get order
        order = self.order_manager.get_order(order_id)
        if not order:
            return AgentResponse(message=f"❌ Order **{order_id}** not found. Please check the order ID.")
        
        # Check if modifiable
        can_modify, reason = self.order_manager.can_modify_order(order_id)
        if not can_modify:
            return AgentResponse(message=f"❌ Cannot modify order {order_id}: {reason}")
        
        # Get changes
        new_size = extracted.get("size")
        new_color = extracted.get("color")
        
        # Build changes
        changes = {}
        changes_desc = []
        
        if new_size and new_size.upper() != order.get('size', '').upper():
            changes['size'] = new_size.upper()
            changes_desc.append(f"Size: {order.get('size')} → {new_size.upper()}")
        
        if new_color and new_color.lower() != order.get('color', '').lower():
            changes['color'] = new_color.capitalize()
            changes_desc.append(f"Color: {order.get('color')} → {new_color.capitalize()}")
        
        if not changes:
            return AgentResponse(
                message=f"""📋 **Order {order_id}**

• Product: {order['product_name']}
• Current Size: {order.get('size', 'N/A')}
• Current Color: {order.get('color', 'N/A')}

What would you like to change? You can update the size or color.

*Example: \"change to size M\" or \"change to black\"*"""
            )
        
        # Store pending action
        state.pending_action = {
            'type': 'modify',
            'order_id': order_id,
            'product_name': order.get('product_name'),
            'old_size': order.get('size'),
            'old_color': order.get('color'),
            'changes': changes,
            'quantity': order.get('quantity', 1)
        }
        
        return AgentResponse(
            message=f"""✏️ **Modify Order {order_id}**

**Current:** {order['product_name']} - {order.get('size')}, {order.get('color')}

**Changes:**
{chr(10).join(['• ' + c for c in changes_desc])}

━━━━━━━━━━━━━━━━━━━━━━
**Type `CHANGE` to confirm modifications**
━━━━━━━━━━━━━━━━━━━━━━""",
            requires_confirmation=True,
            metadata={"pending_action": "modify_order"}
        )
    
    def _handle_cancel_order(self, query: str, state: SharedState, extracted: Dict) -> AgentResponse:
        """Handle order cancellation - supports multiple orders"""
        if not self.order_manager:
            return AgentResponse(message="Order management is currently unavailable.")
        
        order_ids_str = extracted.get("order_id")
        product_mentioned = extracted.get("product_mentioned")
        
        # Parse order IDs
        order_ids = []
        if order_ids_str:
            for oid in order_ids_str.split(","):
                oid = oid.strip().upper()
                if oid and oid not in order_ids:
                    order_ids.append(oid)
        
        # If product mentioned but no order ID
        if not order_ids and product_mentioned:
            user_orders = self.order_manager.get_orders_by_user(state.current_user_id)
            matching = [
                o for o in user_orders
                if product_mentioned.lower() in o.get('product_name', '').lower()
                and self.order_manager.can_cancel_order(o['order_id'])[0]
            ]
            
            if len(matching) == 1:
                order_ids = [matching[0]['order_id']]
            elif len(matching) > 1:
                orders_list = "\n".join([
                    f"• **{o['order_id']}**: {o['product_name']} ({o.get('size')}/{o.get('color')})"
                    for o in matching
                ])
                return AgentResponse(
                    message=f"📋 **Multiple orders found for {product_mentioned}:**\n\n{orders_list}\n\nWhich order would you like to cancel? Please specify the Order ID."
                )
        
        # If still no order IDs, show cancellable orders
        if not order_ids:
            user_orders = self.order_manager.get_orders_by_user(state.current_user_id)
            cancellable = [o for o in user_orders if self.order_manager.can_cancel_order(o['order_id'])[0]]
            
            if cancellable:
                orders_list = "\n".join([
                    f"• **{o['order_id']}**: {o['product_name']} ({o.get('size')}/{o.get('color')}) - {o.get('status', 'N/A').replace('_', ' ').title()}"
                    for o in cancellable[:10]
                ])
                return AgentResponse(
                    message=f"📋 **Orders you can cancel:**\n\n{orders_list}\n\nWhich order would you like to cancel? You can specify multiple orders.\n\n*Example: \"Cancel ORD-12345\" or \"Cancel ORD-12345 and ORD-67890\"*"
                )
            return AgentResponse(message="You don't have any orders that can be cancelled.")
        
        # Validate orders
        valid_orders = []
        invalid_orders = []
        
        for oid in order_ids:
            order = self.order_manager.get_order(oid)
            if not order:
                invalid_orders.append((oid, "Order not found"))
                continue
            
            can_cancel, reason = self.order_manager.can_cancel_order(oid)
            if not can_cancel:
                invalid_orders.append((oid, reason))
                continue
            
            valid_orders.append(order)
        
        if not valid_orders:
            error_msgs = "\n".join([f"• {oid}: {reason}" for oid, reason in invalid_orders])
            return AgentResponse(message=f"❌ **Cannot cancel:**\n\n{error_msgs}")
        
        # Single order
        if len(valid_orders) == 1:
            order = valid_orders[0]
            state.pending_action = {
                'type': 'cancel',
                'order_id': order['order_id']
            }
            
            return AgentResponse(
                message=f"""🗑️ **Cancel Order {order['order_id']}?**

• **Product:** {order['product_name']}
• **Size:** {order.get('size')} | **Color:** {order.get('color')}
• **Price:** {order.get('currency', 'MYR')} {order.get('total_price', 0):.2f}
• **Status:** {order.get('status', 'N/A').replace('_', ' ').title()}

━━━━━━━━━━━━━━━━━━━━━━
**Type `DELETE` to confirm cancellation**
━━━━━━━━━━━━━━━━━━━━━━

_Refund will be processed within 3-5 business days._""",
                requires_confirmation=True,
                metadata={"pending_action": "cancel_order"}
            )
        
        # Multiple orders
        state.pending_action = {
            'type': 'cancel_multiple',
            'order_ids': [o['order_id'] for o in valid_orders]
        }
        
        orders_summary = "\n".join([
            f"• **{o['order_id']}**: {o['product_name']} ({o.get('size')}/{o.get('color')}) - {o.get('currency', 'MYR')} {o.get('total_price', 0):.2f}"
            for o in valid_orders
        ])
        
        total_refund = sum(o.get('total_price', 0) for o in valid_orders)
        
        invalid_msg = ""
        if invalid_orders:
            invalid_msg = "\n\n❌ **Cannot cancel:**\n" + "\n".join([f"• {oid}: {reason}" for oid, reason in invalid_orders])
        
        return AgentResponse(
            message=f"""🗑️ **Cancel {len(valid_orders)} Orders?**

{orders_summary}

**Total Refund:** MYR {total_refund:.2f}{invalid_msg}

━━━━━━━━━━━━━━━━━━━━━━
**Type `DELETE` to confirm cancellation of all orders**
━━━━━━━━━━━━━━━━━━━━━━""",
            requires_confirmation=True,
            metadata={"pending_action": "cancel_multiple"}
        )


# =============================================================================
# CONFIRMATION AGENT - Executes Confirmed Actions
# =============================================================================
class ConfirmationAgent:
    """Executes confirmed actions (ORDER, DELETE, CHANGE)"""
    
    def __init__(self, order_manager=None, user_manager=None):
        self.order_manager = order_manager
        self.user_manager = user_manager
    
    def handle(self, query: str, state: SharedState, extracted: Dict) -> AgentResponse:
        confirm_type = extracted.get("confirm_type", query.strip().upper())
        pending = state.pending_action
        
        if not pending:
            return AgentResponse(
                message="I don't have a pending action to confirm. What would you like to do?"
            )
        
        pending_type = pending.get('type')
        
        if confirm_type == "ORDER" and pending_type == 'create':
            return self._confirm_create_order(state)
        
        if confirm_type == "DELETE" and pending_type in ['cancel', 'cancel_multiple']:
            return self._confirm_cancel_order(state)
        
        if confirm_type == "CHANGE" and pending_type == 'modify':
            return self._confirm_modify_order(state)
        
        return AgentResponse(
            message=f"The confirmation '{confirm_type}' doesn't match the pending action. Please use the correct keyword."
        )
    
    def _confirm_create_order(self, state: SharedState) -> AgentResponse:
        """Execute order creation"""
        if not self.order_manager:
            state.clear_pending_action()
            return AgentResponse(message="Order system is temporarily unavailable.")
        
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
                message=f"""🎉 **Order Confirmed!**

**Order ID:** {order['order_id']}
**Product:** {order['product_name']}
**Size:** {order['size']} | **Color:** {order['color']}
**Total:** {order['currency']} {order['total_price']:.2f}

📧 Confirmation email sent!
📦 Estimated delivery: {order.get('estimated_delivery', '3-7 business days')}

Thank you for shopping with ByNoemie! 💕""",
                action_completed=True,
                metadata={"order_id": order['order_id']}
            )
        except Exception as e:
            state.clear_pending_action()
            return AgentResponse(message=f"❌ Error creating order: {str(e)}")
    
    def _confirm_cancel_order(self, state: SharedState) -> AgentResponse:
        """Execute order cancellation"""
        if not self.order_manager:
            state.clear_pending_action()
            return AgentResponse(message="Order system is temporarily unavailable.")
        
        pending = state.pending_action
        
        # Multiple cancellations
        if pending.get('type') == 'cancel_multiple':
            order_ids = pending.get('order_ids', [])
            results = []
            total_refund = 0
            success_count = 0
            
            for oid in order_ids:
                try:
                    success, message, order = self.order_manager.cancel_order(oid)
                    if success:
                        results.append(f"✅ **{oid}**: Cancelled")
                        total_refund += order.get('total_price', 0)
                        success_count += 1
                    else:
                        results.append(f"❌ **{oid}**: {message}")
                except Exception as e:
                    results.append(f"❌ **{oid}**: Error - {str(e)}")
            
            state.clear_pending_action()
            
            return AgentResponse(
                message=f"""✅ **{success_count} Order(s) Cancelled**

{chr(10).join(results)}

💰 **Total Refund:** MYR {total_refund:.2f}
_Refunds will be processed within 3-5 business days._

Is there anything else I can help you with?""",
                action_completed=True
            )
        
        # Single cancellation
        order_id = pending.get('order_id')
        
        try:
            success, message, order = self.order_manager.cancel_order(order_id)
            state.clear_pending_action()
            
            if success:
                return AgentResponse(
                    message=f"""✅ **Order Cancelled**

Order **{order_id}** has been cancelled successfully.

💰 **Refund:** {order.get('currency', 'MYR')} {order.get('total_price', 0):.2f}
_Will be processed within 3-5 business days._

Is there anything else I can help you with?""",
                    action_completed=True
                )
            else:
                return AgentResponse(message=f"❌ Could not cancel order: {message}")
        except Exception as e:
            state.clear_pending_action()
            return AgentResponse(message=f"❌ Error: {str(e)}")
    
    def _confirm_modify_order(self, state: SharedState) -> AgentResponse:
        """Execute order modification"""
        if not self.order_manager:
            state.clear_pending_action()
            return AgentResponse(message="Order system is temporarily unavailable.")
        
        pending = state.pending_action
        order_id = pending.get('order_id')
        changes = pending.get('changes', {})
        
        try:
            success, message, order = self.order_manager.modify_order_simple(
                order_id=order_id,
                new_size=changes.get('size'),
                new_color=changes.get('color'),
                old_size=pending.get('old_size'),
                old_color=pending.get('old_color')
            )
            state.clear_pending_action()
            
            if success:
                return AgentResponse(
                    message=f"""✅ **Order Modified**

Order **{order_id}** has been updated!

**Changes:** {message}
**Product:** {order.get('product_name')}
**New Details:** Size {order.get('size')} | Color {order.get('color')}

Is there anything else I can help you with?""",
                    action_completed=True
                )
            else:
                return AgentResponse(message=f"❌ Could not modify order: {message}")
        except Exception as e:
            state.clear_pending_action()
            return AgentResponse(message=f"❌ Error: {str(e)}")


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================
class ChatbotOrchestrator:
    """
    Main orchestrator that coordinates all agents.
    Maintains shared state and routes queries through LLM-first router.
    """
    
    def __init__(self, openai_client, products: List[Dict], stock_data: Dict,
                 order_manager=None, user_manager=None, policy_rag=None):
        self.state = SharedState()
        
        product_names = [p['product_name'] for p in products]
        
        # Initialize agents
        self.router = RouterAgent(openai_client, product_names)
        self.deflection_agent = DeflectionAgent(openai_client, products)
        self.info_agent = InfoAgent(openai_client, products, stock_data, order_manager, policy_rag)
        self.action_agent = ActionAgent(openai_client, products, stock_data, order_manager, user_manager)
        self.confirmation_agent = ConfirmationAgent(order_manager, user_manager)
        
        self.agents = {
            AgentType.DEFLECTION: self.deflection_agent,
            AgentType.INFO: self.info_agent,
            AgentType.ACTION: self.action_agent,
            AgentType.CONFIRMATION: self.confirmation_agent
        }
        
        print("🚀 ChatbotOrchestrator initialized with LLM-first routing")
    
    def process(self, query: str, chat_history: List[Dict] = None) -> AgentResponse:
        """Process user query through LLM-first routing"""
        print(f"\n{'='*60}")
        print(f"🤖 Processing: '{query}'")
        
        # Preserve pending action during history sync
        saved_pending = self.state.pending_action
        
        if chat_history:
            self.state.conversation_history = []
            for msg in chat_history:
                self.state.add_message(
                    msg.get("role", "user"),
                    msg.get("content", ""),
                    msg.get("metadata")
                )
        
        if saved_pending and not self.state.pending_action:
            self.state.pending_action = saved_pending
        
        # Add current message
        self.state.add_message("user", query)
        
        # Route query
        agent_type, extracted = self.router.route(query, self.state)
        
        print(f"📌 Agent: {agent_type.value}")
        print(f"   Intent: {extracted.get('intent')}")
        print(f"   Subtype: {extracted.get('action_subtype')}")
        print(f"   Product: {extracted.get('product_mentioned')}")
        print(f"   Order IDs: {extracted.get('order_id')}")
        
        # Execute agent
        agent = self.agents.get(agent_type, self.info_agent)
        response = agent.handle(query, self.state, extracted)
        
        print(f"💬 Response: {response.message[:100]}...")
        print(f"{'='*60}\n")
        
        # Update state
        self.state.add_message("assistant", response.message, {"agent": agent_type.value})
        
        if response.products_to_show:
            if len(response.products_to_show) == 1:
                self.state.set_current_product(response.products_to_show[0])
            self.state.last_shown_products = response.products_to_show
        
        return response
    
    def set_user(self, user_id: str):
        self.state.current_user_id = user_id
    
    def get_state(self) -> SharedState:
        return self.state
    
    def clear_state(self):
        self.state = SharedState()