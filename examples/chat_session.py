"""
Chat Session Example

Demonstrates multi-turn conversation with memory/history management.

Usage:
    export GROQ_API_KEY="your_key"
    python examples/chat_session.py
"""

import os
import sys
from pathlib import Path
from typing import List, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import create_llm_client


class ChatSession:
    """
    Simple chat session with conversation history.
    
    Demonstrates:
    - Maintaining conversation context
    - System prompt customization
    - Message history management
    """
    
    def __init__(self, system_prompt: str = None):
        self.client = create_llm_client()
        self.system_prompt = system_prompt or """You are a helpful fashion assistant for ByNoemie, 
a luxury women's boutique in Malaysia. You help customers find perfect outfits 
for any occasion. Be friendly, knowledgeable, and suggest specific products when relevant."""
        
        self.history: List[Dict[str, str]] = []
        self.max_history = 10  # Keep last N exchanges
    
    def chat(self, user_message: str) -> str:
        """Send a message and get response"""
        
        # Build messages list
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add history
        for exchange in self.history[-self.max_history:]:
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append({"role": "assistant", "content": exchange["assistant"]})
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        # Get response
        response = self.client.generate(messages)
        
        # Store in history
        self.history.append({
            "user": user_message,
            "assistant": response.content
        })
        
        return response.content
    
    def clear_history(self):
        """Clear conversation history"""
        self.history = []
    
    def get_history_summary(self) -> str:
        """Get a summary of the conversation"""
        if not self.history:
            return "No conversation history."
        
        summary = f"Conversation ({len(self.history)} exchanges):\n"
        for i, exchange in enumerate(self.history, 1):
            summary += f"\n{i}. User: {exchange['user'][:50]}..."
            summary += f"\n   Bot: {exchange['assistant'][:50]}..."
        
        return summary


def interactive_session():
    """Run an interactive chat session"""
    print("=" * 60)
    print("ByNoemie Fashion Assistant - Interactive Chat")
    print("=" * 60)
    print("\nType 'quit' to exit, 'clear' to reset, 'history' to see history")
    print("-" * 60)
    
    session = ChatSession()
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("\nGoodbye! 👋")
                break
            
            if user_input.lower() == 'clear':
                session.clear_history()
                print("✨ History cleared!")
                continue
            
            if user_input.lower() == 'history':
                print(session.get_history_summary())
                continue
            
            # Get response
            response = session.chat(user_input)
            print(f"\n🤖 Assistant: {response}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def demo_conversation():
    """Demo a pre-scripted conversation"""
    print("=" * 60)
    print("Demo: Multi-turn Conversation")
    print("=" * 60)
    
    session = ChatSession()
    
    # Pre-scripted conversation
    conversation = [
        "Hi! I'm looking for something for a special occasion.",
        "It's for a romantic dinner date. Any suggestions?",
        "I prefer something in black or red.",
        "Do you have any dresses under MYR 500?",
        "What about shoes to match?",
    ]
    
    for message in conversation:
        print(f"\n👤 User: {message}")
        response = session.chat(message)
        print(f"\n🤖 Assistant: {response}")
        print("-" * 40)
    
    print("\n✅ Demo completed!")
    print(f"\nHistory: {len(session.history)} exchanges")


def demo_context_awareness():
    """Demonstrate context awareness across turns"""
    print("\n" + "=" * 60)
    print("Demo: Context Awareness")
    print("=" * 60)
    
    session = ChatSession(
        system_prompt="You are a fashion assistant. Remember customer preferences."
    )
    
    # Conversation showing context retention
    exchanges = [
        ("My budget is around MYR 400.", None),
        ("I'm attending a beach wedding next month.", None),
        ("What do you recommend based on my budget and event?", None),  # Should remember both
    ]
    
    for user_msg, _ in exchanges:
        print(f"\n👤 User: {user_msg}")
        response = session.chat(user_msg)
        print(f"\n🤖 Assistant: {response}")
        print("-" * 40)


def main():
    """Run examples"""
    print("\n🔥 ByNoemie RAG - Chat Session Examples\n")
    
    # Check for API keys
    has_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if not has_key:
        print("⚠️  Set GROQ_API_KEY or OPENAI_API_KEY first")
        return
    
    # Choose mode
    print("Select mode:")
    print("1. Interactive chat")
    print("2. Demo conversation")
    print("3. Context awareness demo")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        interactive_session()
    elif choice == "2":
        demo_conversation()
    elif choice == "3":
        demo_context_awareness()
    else:
        print("Running demo conversation...")
        demo_conversation()


if __name__ == "__main__":
    main()
