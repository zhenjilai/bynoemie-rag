"""
Basic Completion Example

Demonstrates basic usage of the LLM clients with LangChain integration.

Usage:
    # Set API key
    export GROQ_API_KEY="your_key"  # or OPENAI_API_KEY
    
    # Run
    python examples/basic_completion.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import create_llm_client, LLMClientFactory


def example_basic_chat():
    """Basic chat completion example"""
    print("=" * 60)
    print("Example 1: Basic Chat Completion")
    print("=" * 60)
    
    # Create client (auto-detects available provider)
    client = create_llm_client()
    print(f"Using provider: {client.provider_name}")
    
    # Simple chat
    response = client.chat(
        system_prompt="You are a helpful fashion assistant.",
        user_prompt="What should I wear to a romantic dinner?"
    )
    
    print(f"\nResponse:\n{response.content}")
    print(f"\nTokens used: {response.usage}")
    print(f"Latency: {response.latency_ms:.0f}ms")


def example_specific_provider():
    """Using a specific provider"""
    print("\n" + "=" * 60)
    print("Example 2: Specific Provider (Groq)")
    print("=" * 60)
    
    try:
        # Create Groq client specifically
        client = LLMClientFactory.create(
            "groq",
            model="llama-3.1-70b-versatile",
            temperature=0.7
        )
        
        if not client.is_available():
            print("Groq not available - set GROQ_API_KEY")
            return
        
        response = client.chat(
            system_prompt="You are a creative fashion stylist.",
            user_prompt="Describe the vibe of a sequin mini dress in 3 words."
        )
        
        print(f"\nResponse: {response.content}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_langchain_integration():
    """Using LangChain model"""
    print("\n" + "=" * 60)
    print("Example 3: LangChain Integration")
    print("=" * 60)
    
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    # Get LangChain compatible model
    client = create_llm_client()
    llm = client.get_langchain_model()
    
    # Create chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a fashion expert. Be concise."),
        ("human", "What vibe does {color} {item} give off?")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    # Run chain
    result = chain.invoke({
        "color": "black",
        "item": "sequin dress"
    })
    
    print(f"\nResult: {result}")


def example_json_output():
    """Getting structured JSON output"""
    print("\n" + "=" * 60)
    print("Example 4: JSON Output")
    print("=" * 60)
    
    from src.llm import parse_json_response
    
    client = create_llm_client()
    
    response = client.chat(
        system_prompt="You are a fashion analyst. Return only valid JSON.",
        user_prompt="""Analyze this product:
        
Name: Coco Dress
Description: An ultra-mini silhouette covered in black sequins.

Return JSON with: {"vibe_tags": [...], "primary_occasion": "..."}"""
    )
    
    try:
        data = parse_json_response(response.content)
        print(f"\nParsed JSON:")
        print(f"  Vibe tags: {data.get('vibe_tags', [])}")
        print(f"  Occasion: {data.get('primary_occasion', 'N/A')}")
    except Exception as e:
        print(f"Raw response: {response.content}")
        print(f"Parse error: {e}")


def example_streaming():
    """Streaming response example (if supported)"""
    print("\n" + "=" * 60)
    print("Example 5: Streaming (LangChain)")
    print("=" * 60)
    
    client = create_llm_client()
    llm = client.get_langchain_model()
    
    print("\nStreaming response:")
    for chunk in llm.stream("Write a 3-line poem about a red dress"):
        print(chunk.content, end="", flush=True)
    print("\n")


def main():
    """Run all examples"""
    print("\n🔥 ByNoemie RAG - Basic Completion Examples\n")
    
    # Check for API keys
    has_key = any([
        os.getenv("GROQ_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY")
    ])
    
    if not has_key:
        print("⚠️  No API keys found. Set one of:")
        print("   - GROQ_API_KEY (free at https://console.groq.com)")
        print("   - OPENAI_API_KEY")
        print("   - ANTHROPIC_API_KEY")
        print("\nExample: export GROQ_API_KEY='your_key_here'")
        return
    
    try:
        example_basic_chat()
        example_specific_provider()
        example_langchain_integration()
        example_json_output()
        example_streaming()
        
        print("\n✅ All examples completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
