#!/usr/bin/env python3
"""
Complete Workflow Demo

Demonstrates the full pipeline:
1. Load products from CSV
2. Store in ChromaDB
3. Generate vibes (only for new products)
4. Search products by query or vibe

Usage:
    # First time: Process all products
    python scripts/process_products.py --csv data/products/sample_products.csv
    
    # Add new products: Only processes new ones
    python scripts/process_products.py --csv data/products/updated_products.csv
    
    # Force regenerate all vibes
    python scripts/process_products.py --csv data/products/sample_products.csv --force
    
    # Search products
    python scripts/process_products.py --search "romantic dinner dress"
    
    # View database stats
    python scripts/process_products.py --stats
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_environment():
    """Load API keys from .env file"""
    env_file = PROJECT_ROOT / ".env"
    
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            logger.info(f"✅ Loaded environment from {env_file}")
        except ImportError:
            logger.warning("python-dotenv not installed, using manual .env parsing")
            # Manual parsing
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, _, value = line.partition('=')
                        os.environ[key.strip()] = value.strip().strip('"').strip("'")
    else:
        logger.warning(f"⚠️  No .env file found at {env_file}")
        logger.info("   Copy .env.example to .env and add your API keys")


def process_csv(args):
    """Process CSV file and generate vibes"""
    from src.rag import ProductDatabase, DataProcessor
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return
    
    # Initialize database
    db = ProductDatabase(persist_directory=args.db_path)
    
    # Initialize processor
    processor = DataProcessor(
        database=db,
        vibe_method=args.method
    )
    
    # Process CSV
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {csv_path}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Force regenerate: {args.force}")
    logger.info(f"{'='*60}\n")
    
    stats = processor.process_csv(
        csv_path=str(csv_path),
        force_regenerate=args.force
    )
    
    # Export if requested
    if args.export:
        processor.export_to_json(args.export)
        logger.info(f"✅ Exported to {args.export}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 Processing Summary")
    print(f"{'='*60}")
    print(f"  Total products:     {stats.total_products}")
    print(f"  New products:       {stats.new_products}")
    print(f"  Updated products:   {stats.updated_products}")
    print(f"  Unchanged products: {stats.unchanged_products}")
    print(f"  Vibes generated:    {stats.vibes_generated}")
    print(f"  Vibes skipped:      {stats.vibes_skipped}")
    print(f"  Errors:             {stats.errors}")
    print(f"  Processing time:    {stats.processing_time_seconds:.2f}s")
    print(f"{'='*60}\n")


def search_products(args):
    """Search products by query"""
    from src.rag import ProductDatabase
    
    db = ProductDatabase(persist_directory=args.db_path)
    
    query = args.search
    logger.info(f"\n🔍 Searching for: '{query}'\n")
    
    # Combined search (products + vibes)
    results = db.search(query, n_results=5)
    
    if not results:
        print("No results found.")
        return
    
    print(f"{'='*60}")
    print(f"Found {len(results)} results:")
    print(f"{'='*60}\n")
    
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.get('product_name', 'Unknown')}")
        print(f"   Type: {r.get('product_type', 'N/A')}")
        print(f"   Price: {r.get('price_currency', 'MYR')} {r.get('price_min', 0)}")
        print(f"   Vibes: {', '.join(r.get('vibe_tags', [])[:5])}")
        print(f"   Score: {r.get('combined_score', 0):.3f}")
        print(f"   Mood: {r.get('mood_summary', 'N/A')[:80]}")
        print()


def show_stats(args):
    """Show database statistics"""
    from src.rag import ProductDatabase
    
    db = ProductDatabase(persist_directory=args.db_path)
    stats = db.get_stats()
    
    print(f"\n{'='*60}")
    print("📊 Database Statistics")
    print(f"{'='*60}")
    print(f"  Products count:        {stats['products_count']}")
    print(f"  Vibes count:           {stats['vibes_count']}")
    print(f"  Products without vibes: {stats['products_without_vibes']}")
    print(f"  Database path:         {stats['persist_directory']}")
    print(f"{'='*60}\n")
    
    # Show sample products
    products = db.get_all_products()[:3]
    if products:
        print("Sample products:")
        for p in products:
            vibes = db.get_vibes(p['product_id'])
            print(f"  - {p['product_name']}")
            if vibes:
                print(f"    Vibes: {', '.join(vibes['vibe_tags'][:3])}...")


def interactive_demo(args):
    """Interactive demo mode"""
    from src.rag import ProductDatabase
    
    db = ProductDatabase(persist_directory=args.db_path)
    
    print(f"\n{'='*60}")
    print("🎯 ByNoemie Product Search - Interactive Demo")
    print(f"{'='*60}")
    print("Type a search query (or 'quit' to exit)")
    print("Examples: 'romantic dinner', 'night out', 'elegant dress'")
    print(f"{'='*60}\n")
    
    while True:
        try:
            query = input("\n🔍 Search: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ('quit', 'exit', 'q'):
                print("Goodbye! 👋")
                break
            
            results = db.search(query, n_results=3)
            
            if not results:
                print("No results found. Try a different query.")
                continue
            
            print(f"\nTop {len(results)} results:\n")
            
            for i, r in enumerate(results, 1):
                print(f"  {i}. {r.get('product_name', 'Unknown')}")
                print(f"     {', '.join(r.get('vibe_tags', [])[:4])}")
                print(f"     Score: {r.get('combined_score', 0):.2f}")
                print()
                
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break


def main():
    parser = argparse.ArgumentParser(
        description="ByNoemie Product Data Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process CSV:          python scripts/process_products.py --csv products.csv
  Search products:      python scripts/process_products.py --search "romantic dinner"
  View stats:           python scripts/process_products.py --stats
  Interactive mode:     python scripts/process_products.py --interactive
  Force regenerate:     python scripts/process_products.py --csv products.csv --force
        """
    )
    
    # Input options
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to products CSV file"
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Search query"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive search mode"
    )
    
    # Processing options
    parser.add_argument(
        "--method",
        choices=["rule_based", "llm", "hybrid"],
        default="hybrid",
        help="Vibe generation method (default: hybrid)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regenerate vibes for all products"
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export results to JSON file"
    )
    
    # Database options
    parser.add_argument(
        "--db-path",
        type=str,
        default="./data/embeddings/chroma_db",
        help="ChromaDB persist directory"
    )
    
    args = parser.parse_args()
    
    # Load environment
    load_environment()
    
    # Execute requested action
    if args.csv:
        process_csv(args)
    elif args.search:
        search_products(args)
    elif args.stats:
        show_stats(args)
    elif args.interactive:
        interactive_demo(args)
    else:
        parser.print_help()
        print("\n💡 Quick start:")
        print("   python scripts/process_products.py --csv data/products/sample_products.csv")


if __name__ == "__main__":
    main()
