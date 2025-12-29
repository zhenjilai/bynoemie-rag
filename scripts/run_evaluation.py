#!/usr/bin/env python3
"""
RAG Evaluation Runner

Evaluates the ByNoemie RAG system using standard metrics.

Usage:
    # Run basic evaluation (no LLM needed)
    python scripts/run_evaluation.py
    
    # Run with LLM-based answer evaluation
    python scripts/run_evaluation.py --with-llm
    
    # Export results
    python scripts/run_evaluation.py --export results.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_environment():
    """Load .env file"""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            pass


def run_retrieval_evaluation():
    """Run retrieval-only evaluation"""
    from src.evaluation import RetrievalEvaluator, BYNOEMIE_TEST_CASES
    from src.rag import ProductDatabase
    
    print("\n" + "="*60)
    print("🔍 Retrieval Quality Evaluation")
    print("="*60)
    
    # Initialize database
    print("\n📦 Loading database...")
    db = ProductDatabase()
    
    stats = db.get_stats()
    print(f"   Products: {stats['products_count']}")
    print(f"   Vibes: {stats['vibes_count']}")
    
    if stats['products_count'] == 0:
        print("\n⚠️  Database is empty! Run process_products.py first:")
        print("   python scripts/process_products.py --csv data/products/sample_products.csv")
        return None
    
    # Run evaluation
    evaluator = RetrievalEvaluator()
    
    retrieved_results = []
    ground_truth = []
    queries = []
    
    print("\n🔍 Running search queries...")
    
    for i, case in enumerate(BYNOEMIE_TEST_CASES):
        query = case["query"]
        expected_ids = case["ground_truth_ids"]
        
        print(f"   [{i+1}/{len(BYNOEMIE_TEST_CASES)}] {query}")
        
        # Search using the database
        results = db.search(query, n_results=10)
        
        # Extract IDs
        result_ids = [r.get("product_id", "") for r in results]
        
        retrieved_results.append(result_ids)
        ground_truth.append(expected_ids)
        queries.append(query)
    
    # Calculate metrics
    print("\n📊 Calculating metrics...")
    
    metrics, per_query = evaluator.evaluate_batch(
        retrieved_results=retrieved_results,
        ground_truth=ground_truth,
        queries=queries
    )
    
    # Print results
    print("\n" + "="*60)
    print("📈 RETRIEVAL METRICS")
    print("="*60)
    
    print(f"\n{'Metric':<20} {'Value':>10}")
    print("-"*32)
    print(f"{'Recall@1':<20} {metrics.recall_at_1:>10.3f}")
    print(f"{'Recall@3':<20} {metrics.recall_at_3:>10.3f}")
    print(f"{'Recall@5':<20} {metrics.recall_at_5:>10.3f}")
    print(f"{'Recall@10':<20} {metrics.recall_at_10:>10.3f}")
    print("-"*32)
    print(f"{'Precision@1':<20} {metrics.precision_at_1:>10.3f}")
    print(f"{'Precision@5':<20} {metrics.precision_at_5:>10.3f}")
    print("-"*32)
    print(f"{'MRR':<20} {metrics.mrr:>10.3f}")
    print(f"{'NDCG@5':<20} {metrics.ndcg_at_5:>10.3f}")
    print(f"{'NDCG@10':<20} {metrics.ndcg_at_10:>10.3f}")
    print("-"*32)
    print(f"{'Hit Rate@1':<20} {metrics.hit_rate_at_1:>10.3f}")
    print(f"{'Hit Rate@5':<20} {metrics.hit_rate_at_5:>10.3f}")
    print(f"{'MAP':<20} {metrics.map_score:>10.3f}")
    
    # Per-query breakdown
    print("\n" + "="*60)
    print("📝 PER-QUERY RESULTS")
    print("="*60)
    print(f"\n{'Query':<35} {'Hit@5':>8} {'Recall@5':>10} {'MRR':>8}")
    print("-"*65)
    
    for r in per_query:
        hit = "✅" if r.retrieval_metrics['hit_rate_at_5'] > 0 else "❌"
        print(f"{r.query[:33]:<35} {hit:>8} {r.retrieval_metrics['recall_at_5']:>10.2f} {r.retrieval_metrics['mrr']:>8.2f}")
    
    return {
        "metrics": metrics.to_dict(),
        "per_query": [{"query": r.query, **r.retrieval_metrics} for r in per_query]
    }


def run_full_rag_evaluation(with_llm: bool = False):
    """Run full RAG evaluation including answer quality"""
    from src.evaluation import RAGEvaluator, BYNOEMIE_TEST_CASES
    from src.rag import ProductDatabase
    
    print("\n" + "="*60)
    print("🎯 Full RAG System Evaluation")
    print("="*60)
    
    # Initialize
    db = ProductDatabase()
    
    # Setup LLM client if requested
    llm_client = None
    if with_llm:
        try:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY", "")
            if api_key:
                llm_client = Groq(api_key=api_key)
                print("✅ LLM client initialized for answer evaluation")
            else:
                print("⚠️  No GROQ_API_KEY found, using rule-based evaluation")
        except ImportError:
            print("⚠️  groq not installed, using rule-based evaluation")
    
    evaluator = RAGEvaluator(llm_client=llm_client)
    
    # Define RAG function
    def rag_function(query: str) -> dict:
        results = db.search(query, n_results=5)
        
        retrieved_ids = [r.get("product_id", "") for r in results]
        
        # Build context from results
        context_parts = []
        for r in results:
            context_parts.append(
                f"{r.get('product_name', 'Unknown')}: "
                f"{', '.join(r.get('vibe_tags', [])[:5])}"
            )
        context = "\n".join(context_parts)
        
        # Simple answer (in real system, this would be LLM-generated)
        if results:
            top = results[0]
            answer = f"I recommend the {top.get('product_name', 'Unknown')} for {query}."
        else:
            answer = "I couldn't find a matching product."
        
        return {
            "retrieved_ids": retrieved_ids,
            "context": context,
            "answer": answer
        }
    
    # Run evaluation
    results = evaluator.evaluate_rag_system(
        rag_function=rag_function,
        test_cases=BYNOEMIE_TEST_CASES,
        verbose=True
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Runner")
    
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Use LLM for answer quality evaluation"
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export results to JSON file"
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Run only retrieval evaluation"
    )
    
    args = parser.parse_args()
    
    # Load environment
    load_environment()
    
    # Run evaluation
    if args.retrieval_only:
        results = run_retrieval_evaluation()
    else:
        results = run_full_rag_evaluation(with_llm=args.with_llm)
    
    # Export if requested
    if args.export and results:
        with open(args.export, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✅ Results exported to {args.export}")
    
    # Summary
    print("\n" + "="*60)
    print("📋 Evaluation Complete!")
    print("="*60)
    
    if results and "retrieval_metrics" in results:
        r = results["retrieval_metrics"]
        print(f"\n🎯 Key Metrics:")
        print(f"   Recall@5:    {r.get('recall_at_5', 0):.1%}")
        print(f"   MRR:         {r.get('mrr', 0):.1%}")
        print(f"   Hit Rate@5:  {r.get('hit_rate_at_5', 0):.1%}")


if __name__ == "__main__":
    main()
