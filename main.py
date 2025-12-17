"""
ByNoemie RAG Chatbot - Main Entry Point

Usage:
    # Interactive chat
    python main.py --mode chat
    
    # Process products (generate vibes)
    python main.py --mode process --csv products.csv
    
    # Start API server
    python main.py --mode serve
    
    # Run examples
    python main.py --mode examples
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Setup environment and check prerequisites"""
    # Check for API keys
    has_key = any([
        os.getenv("GROQ_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY")
    ])
    
    if not has_key:
        logger.warning("No LLM API key found!")
        logger.info("Set one of: GROQ_API_KEY (free), OPENAI_API_KEY, ANTHROPIC_API_KEY")
        logger.info("Get free Groq key at: https://console.groq.com")
    
    # Setup LangSmith if available
    if os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ.setdefault("LANGCHAIN_PROJECT", "bynoemie-rag")
        logger.info("LangSmith tracing enabled")


def run_chat():
    """Run interactive chat mode"""
    from examples.chat_session import interactive_session
    interactive_session()


def run_process(csv_path: str, output_dir: str = "./data"):
    """Process products and generate vibes"""
    import csv
    import json
    from pathlib import Path
    
    logger.info(f"Processing products from: {csv_path}")
    
    # Load products
    products = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        products = list(reader)
    
    logger.info(f"Loaded {len(products)} products")
    
    # Try LangGraph workflow first, fall back to rule-based
    try:
        from src.vibe_generator import create_vibe_generator
        
        generator = create_vibe_generator()
        results = generator.generate_batch(products)
        method = "langgraph"
        
    except Exception as e:
        logger.warning(f"LangGraph failed: {e}, using rule-based")
        from src.vibe_generator import process_products_batch
        
        results = process_products_batch(products)
        method = "rule_based"
    
    # Save results
    output_path = Path(output_dir) / "products"
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / "products_with_vibes.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(results)} products to {output_file}")
    logger.info(f"Method used: {method}")
    
    # Print sample
    print("\n📋 Sample Results:")
    for r in results[:3]:
        name = r.get('product_name', 'Unknown')
        vibes = r.get('vibe_tags', [])[:5]
        print(f"  • {name}: {', '.join(vibes)}")


def run_serve(host: str = "0.0.0.0", port: int = 8000):
    """Start API server"""
    try:
        import uvicorn
        from fastapi import FastAPI
        
        app = FastAPI(
            title="ByNoemie RAG Chatbot API",
            version="1.0.0"
        )
        
        @app.get("/health")
        def health():
            return {"status": "healthy"}
        
        @app.get("/")
        def root():
            return {
                "name": "ByNoemie RAG Chatbot",
                "version": "1.0.0",
                "docs": "/docs"
            }
        
        logger.info(f"Starting server at http://{host}:{port}")
        uvicorn.run(app, host=host, port=port)
        
    except ImportError:
        logger.error("FastAPI/Uvicorn not installed")
        logger.info("Install with: pip install fastapi uvicorn")


def run_examples():
    """Run example scripts"""
    print("\n🔥 ByNoemie RAG - Running Examples\n")
    
    print("Select example to run:")
    print("1. Basic completion")
    print("2. Chat session")
    print("3. Prompt chaining")
    print("4. Vibe generation")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        from examples.basic_completion import main
        main()
    elif choice == "2":
        from examples.chat_session import main
        main()
    elif choice == "3":
        from examples.chain_prompts import main
        main()
    elif choice == "4":
        run_vibe_demo()
    else:
        print("Invalid choice")


def run_vibe_demo():
    """Demo vibe generation"""
    print("\n🏷️ Vibe Generation Demo\n")
    
    # Sample product
    product = {
        "product_name": "Coco Dress",
        "product_type": "Dress",
        "product_description": "All eyes on you in the Coco Dress, an ultra-mini silhouette covered in oversized black sequins. Featuring slim straps and a daring open back, it's made for nights that sparkle.",
        "colors_available": "Black, Gold",
        "material": "Sequin"
    }
    
    print(f"Product: {product['product_name']}")
    print(f"Description: {product['product_description'][:100]}...")
    print("-" * 50)
    
    # Rule-based
    from src.vibe_generator import extract_vibes_from_product
    rule_vibes = extract_vibes_from_product(product)
    print(f"\n1. Rule-based vibes: {rule_vibes}")
    
    # LangGraph (if available)
    try:
        from src.vibe_generator import create_vibe_generator
        
        generator = create_vibe_generator()
        result = generator.generate(
            product_name=product["product_name"],
            product_type=product["product_type"],
            description=product["product_description"],
            colors=product["colors_available"],
            material=product["material"]
        )
        
        print(f"\n2. LangGraph vibes: {result['vibe_tags']}")
        if result.get('mood_summary'):
            print(f"   Mood: {result['mood_summary']}")
            
    except Exception as e:
        print(f"\n2. LangGraph not available: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="ByNoemie RAG Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --mode chat
    python main.py --mode process --csv products.csv
    python main.py --mode serve --port 8000
    python main.py --mode examples
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["chat", "process", "serve", "examples"],
        default="examples",
        help="Operation mode"
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="CSV file for process mode"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data",
        help="Output directory"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_environment()
    
    # Run mode
    if args.mode == "chat":
        run_chat()
    elif args.mode == "process":
        if not args.csv:
            logger.error("--csv required for process mode")
            sys.exit(1)
        run_process(args.csv, args.output)
    elif args.mode == "serve":
        run_serve(args.host, args.port)
    elif args.mode == "examples":
        run_examples()


if __name__ == "__main__":
    main()
