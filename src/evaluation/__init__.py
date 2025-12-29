"""
RAG Evaluation Metrics

Provides evaluation metrics for:
1. Retrieval Quality - How well does the retriever find relevant documents?
2. Answer Relevance - How relevant/accurate are the generated answers?
3. End-to-End RAG Quality - Overall system performance

Metrics implemented:
- Recall@K, Precision@K, MRR, NDCG, Hit Rate (Retrieval)
- Faithfulness, Answer Relevance, Context Relevance (Generation)
- RAGAS-style metrics using LLM-as-judge

Usage:
    from src.evaluation import RetrievalEvaluator, RAGEvaluator
    
    # Evaluate retrieval
    evaluator = RetrievalEvaluator()
    metrics = evaluator.evaluate(queries, retrieved_docs, ground_truth)
    
    # Evaluate full RAG
    rag_eval = RAGEvaluator()
    results = rag_eval.evaluate(queries, contexts, answers, ground_truth)
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics"""
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg_at_5: float = 0.0  # Normalized Discounted Cumulative Gain
    ndcg_at_10: float = 0.0
    hit_rate_at_1: float = 0.0
    hit_rate_at_5: float = 0.0
    map_score: float = 0.0  # Mean Average Precision
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
    
    def __str__(self) -> str:
        return (
            f"Retrieval Metrics:\n"
            f"  Recall@5: {self.recall_at_5:.3f}\n"
            f"  Precision@5: {self.precision_at_5:.3f}\n"
            f"  MRR: {self.mrr:.3f}\n"
            f"  NDCG@5: {self.ndcg_at_5:.3f}\n"
            f"  Hit Rate@5: {self.hit_rate_at_5:.3f}"
        )


@dataclass
class AnswerMetrics:
    """Answer/Generation evaluation metrics"""
    faithfulness: float = 0.0  # Is answer grounded in context?
    answer_relevance: float = 0.0  # Does answer address the query?
    context_relevance: float = 0.0  # Is retrieved context relevant?
    context_utilization: float = 0.0  # How much context is used?
    answer_completeness: float = 0.0  # Does answer cover all aspects?
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
    
    def __str__(self) -> str:
        return (
            f"Answer Metrics:\n"
            f"  Faithfulness: {self.faithfulness:.3f}\n"
            f"  Answer Relevance: {self.answer_relevance:.3f}\n"
            f"  Context Relevance: {self.context_relevance:.3f}"
        )


@dataclass
class EvaluationResult:
    """Complete evaluation result"""
    query: str
    retrieved_ids: List[str]
    ground_truth_ids: List[str]
    answer: str = ""
    context: str = ""
    retrieval_metrics: Dict[str, float] = field(default_factory=dict)
    answer_metrics: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# RETRIEVAL EVALUATOR
# =============================================================================

class RetrievalEvaluator:
    """
    Evaluates retrieval quality using standard IR metrics.
    
    Usage:
        evaluator = RetrievalEvaluator()
        
        # Single query evaluation
        metrics = evaluator.evaluate_single(
            retrieved_ids=["prod1", "prod2", "prod3"],
            relevant_ids=["prod1", "prod4"]
        )
        
        # Batch evaluation
        results = evaluator.evaluate_batch(
            queries=["romantic dinner", "night out"],
            retrieved_results=[["prod1", "prod2"], ["prod3", "prod4"]],
            ground_truth=[["prod1"], ["prod3", "prod5"]]
        )
    """
    
    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        self.k_values = k_values
    
    def recall_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """
        Recall@K: What fraction of relevant items are in top-K results?
        
        Formula: |relevant ∩ retrieved@K| / |relevant|
        """
        if not relevant_ids:
            return 0.0
        
        retrieved_at_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        hits = len(retrieved_at_k & relevant_set)
        return hits / len(relevant_set)
    
    def precision_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """
        Precision@K: What fraction of top-K results are relevant?
        
        Formula: |relevant ∩ retrieved@K| / K
        """
        if k == 0:
            return 0.0
        
        retrieved_at_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        hits = len(retrieved_at_k & relevant_set)
        return hits / min(k, len(retrieved_ids)) if retrieved_ids else 0.0
    
    def mrr(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """
        Mean Reciprocal Rank: 1 / rank of first relevant result
        
        MRR rewards systems that rank relevant items higher.
        """
        relevant_set = set(relevant_ids)
        
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def ndcg_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """
        Normalized Discounted Cumulative Gain@K
        
        NDCG considers the position of relevant items (higher is better).
        Uses binary relevance (1 if relevant, 0 if not).
        """
        relevant_set = set(relevant_ids)
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            if doc_id in relevant_set:
                # Binary relevance, log2(i+2) for discount
                dcg += 1.0 / math.log2(i + 2)
        
        # Calculate ideal DCG (all relevant items at top)
        idcg = 0.0
        for i in range(min(len(relevant_ids), k)):
            idcg += 1.0 / math.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def hit_rate_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """
        Hit Rate@K: Is there at least one relevant item in top-K?
        
        Returns 1.0 if hit, 0.0 if miss.
        """
        retrieved_at_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        return 1.0 if (retrieved_at_k & relevant_set) else 0.0
    
    def average_precision(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """
        Average Precision: Average of precision@k for each relevant item found
        
        AP rewards relevant items appearing earlier in the list.
        """
        relevant_set = set(relevant_ids)
        
        if not relevant_set:
            return 0.0
        
        precisions = []
        relevant_count = 0
        
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                relevant_count += 1
                precision = relevant_count / (i + 1)
                precisions.append(precision)
        
        if not precisions:
            return 0.0
        
        return sum(precisions) / len(relevant_set)
    
    def evaluate_single(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> RetrievalMetrics:
        """Evaluate a single query's retrieval results"""
        
        metrics = RetrievalMetrics(
            recall_at_1=self.recall_at_k(retrieved_ids, relevant_ids, 1),
            recall_at_3=self.recall_at_k(retrieved_ids, relevant_ids, 3),
            recall_at_5=self.recall_at_k(retrieved_ids, relevant_ids, 5),
            recall_at_10=self.recall_at_k(retrieved_ids, relevant_ids, 10),
            precision_at_1=self.precision_at_k(retrieved_ids, relevant_ids, 1),
            precision_at_3=self.precision_at_k(retrieved_ids, relevant_ids, 3),
            precision_at_5=self.precision_at_k(retrieved_ids, relevant_ids, 5),
            precision_at_10=self.precision_at_k(retrieved_ids, relevant_ids, 10),
            mrr=self.mrr(retrieved_ids, relevant_ids),
            ndcg_at_5=self.ndcg_at_k(retrieved_ids, relevant_ids, 5),
            ndcg_at_10=self.ndcg_at_k(retrieved_ids, relevant_ids, 10),
            hit_rate_at_1=self.hit_rate_at_k(retrieved_ids, relevant_ids, 1),
            hit_rate_at_5=self.hit_rate_at_k(retrieved_ids, relevant_ids, 5),
            map_score=self.average_precision(retrieved_ids, relevant_ids)
        )
        
        return metrics
    
    def evaluate_batch(
        self,
        retrieved_results: List[List[str]],
        ground_truth: List[List[str]],
        queries: List[str] = None
    ) -> Tuple[RetrievalMetrics, List[EvaluationResult]]:
        """
        Evaluate multiple queries and return aggregated metrics.
        
        Args:
            retrieved_results: List of retrieved doc IDs per query
            ground_truth: List of relevant doc IDs per query
            queries: Optional list of query strings
            
        Returns:
            Tuple of (aggregated_metrics, per_query_results)
        """
        if len(retrieved_results) != len(ground_truth):
            raise ValueError("retrieved_results and ground_truth must have same length")
        
        all_metrics = defaultdict(list)
        results = []
        
        for i, (retrieved, relevant) in enumerate(zip(retrieved_results, ground_truth)):
            metrics = self.evaluate_single(retrieved, relevant)
            
            # Collect for averaging
            for key, value in metrics.to_dict().items():
                all_metrics[key].append(value)
            
            # Store per-query result
            result = EvaluationResult(
                query=queries[i] if queries else f"query_{i}",
                retrieved_ids=retrieved,
                ground_truth_ids=relevant,
                retrieval_metrics=metrics.to_dict()
            )
            results.append(result)
        
        # Average metrics
        avg_metrics = RetrievalMetrics(**{
            key: sum(values) / len(values) if values else 0.0
            for key, values in all_metrics.items()
        })
        
        return avg_metrics, results


# =============================================================================
# ANSWER EVALUATOR (LLM-as-Judge)
# =============================================================================

class AnswerEvaluator:
    """
    Evaluates answer quality using LLM-as-judge approach.
    
    Implements RAGAS-style metrics:
    - Faithfulness: Is the answer grounded in the provided context?
    - Answer Relevance: Does the answer address the query?
    - Context Relevance: Is the retrieved context relevant to the query?
    
    Usage:
        evaluator = AnswerEvaluator(llm_client)
        
        metrics = evaluator.evaluate(
            query="What dress for romantic dinner?",
            context="Tiara Dress is elegant with satin fabric...",
            answer="I recommend the Tiara Dress for a romantic dinner."
        )
    """
    
    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: LLM client for evaluation. If None, uses rule-based fallback.
        """
        self.llm_client = llm_client
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM for evaluation"""
        if self.llm_client is None:
            return ""
        
        try:
            response = self.llm_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            return ""
    
    def evaluate_faithfulness(
        self,
        answer: str,
        context: str
    ) -> float:
        """
        Faithfulness: Is the answer grounded in the context?
        
        Score 0-1: 1.0 = fully grounded, 0.0 = hallucinated
        """
        if not answer or not context:
            return 0.0
        
        if self.llm_client:
            prompt = f"""Evaluate if the answer is factually grounded in the context.
Score from 0 to 1 where:
- 1.0 = Answer is fully supported by context
- 0.5 = Answer is partially supported
- 0.0 = Answer contains information not in context (hallucination)

Context: {context[:1000]}

Answer: {answer}

Return ONLY a number between 0 and 1:"""
            
            response = self._call_llm(prompt)
            try:
                score = float(response.split()[0])
                return max(0.0, min(1.0, score))
            except:
                pass
        
        # Fallback: Simple word overlap
        context_words = set(context.lower().split())
        answer_words = set(answer.lower().split())
        
        if not answer_words:
            return 0.0
        
        overlap = len(answer_words & context_words)
        return min(1.0, overlap / len(answer_words))
    
    def evaluate_answer_relevance(
        self,
        query: str,
        answer: str
    ) -> float:
        """
        Answer Relevance: Does the answer address the query?
        
        Score 0-1: 1.0 = fully relevant, 0.0 = off-topic
        """
        if not query or not answer:
            return 0.0
        
        if self.llm_client:
            prompt = f"""Evaluate if the answer addresses the user's query.
Score from 0 to 1 where:
- 1.0 = Answer directly and completely addresses the query
- 0.5 = Answer partially addresses the query
- 0.0 = Answer is off-topic or doesn't address the query

Query: {query}

Answer: {answer}

Return ONLY a number between 0 and 1:"""
            
            response = self._call_llm(prompt)
            try:
                score = float(response.split()[0])
                return max(0.0, min(1.0, score))
            except:
                pass
        
        # Fallback: Keyword overlap
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        overlap = len(query_words & answer_words)
        return min(1.0, overlap / max(len(query_words), 1))
    
    def evaluate_context_relevance(
        self,
        query: str,
        context: str
    ) -> float:
        """
        Context Relevance: Is the retrieved context relevant to the query?
        
        Score 0-1: 1.0 = highly relevant, 0.0 = irrelevant
        """
        if not query or not context:
            return 0.0
        
        if self.llm_client:
            prompt = f"""Evaluate if the context is relevant to answering the query.
Score from 0 to 1 where:
- 1.0 = Context is highly relevant and useful
- 0.5 = Context is somewhat relevant
- 0.0 = Context is irrelevant to the query

Query: {query}

Context: {context[:1000]}

Return ONLY a number between 0 and 1:"""
            
            response = self._call_llm(prompt)
            try:
                score = float(response.split()[0])
                return max(0.0, min(1.0, score))
            except:
                pass
        
        # Fallback: Keyword overlap
        query_words = set(query.lower().split())
        context_words = set(context.lower().split())
        
        overlap = len(query_words & context_words)
        return min(1.0, overlap / max(len(query_words), 1))
    
    def evaluate(
        self,
        query: str,
        context: str,
        answer: str
    ) -> AnswerMetrics:
        """Evaluate all answer metrics"""
        
        return AnswerMetrics(
            faithfulness=self.evaluate_faithfulness(answer, context),
            answer_relevance=self.evaluate_answer_relevance(query, answer),
            context_relevance=self.evaluate_context_relevance(query, context)
        )


# =============================================================================
# COMBINED RAG EVALUATOR
# =============================================================================

class RAGEvaluator:
    """
    End-to-end RAG evaluation combining retrieval and answer metrics.
    
    Usage:
        evaluator = RAGEvaluator(llm_client=groq_client)
        
        # Create test cases
        test_cases = [
            {
                "query": "romantic dinner dress",
                "ground_truth_ids": ["PROD002"],  # Tiara Dress
            },
            {
                "query": "night out sparkle",
                "ground_truth_ids": ["PROD001"],  # Coco Dress
            }
        ]
        
        # Run evaluation
        results = evaluator.evaluate_rag_system(
            rag_function=my_rag_function,
            test_cases=test_cases
        )
    """
    
    def __init__(self, llm_client=None):
        self.retrieval_evaluator = RetrievalEvaluator()
        self.answer_evaluator = AnswerEvaluator(llm_client)
    
    def evaluate_single(
        self,
        query: str,
        retrieved_ids: List[str],
        ground_truth_ids: List[str],
        context: str = "",
        answer: str = ""
    ) -> Dict[str, Any]:
        """Evaluate a single RAG result"""
        
        # Retrieval metrics
        retrieval_metrics = self.retrieval_evaluator.evaluate_single(
            retrieved_ids, ground_truth_ids
        )
        
        # Answer metrics (if answer provided)
        answer_metrics = None
        if answer and context:
            answer_metrics = self.answer_evaluator.evaluate(
                query, context, answer
            )
        
        return {
            "query": query,
            "retrieval": retrieval_metrics.to_dict(),
            "answer": answer_metrics.to_dict() if answer_metrics else None
        }
    
    def evaluate_rag_system(
        self,
        rag_function,
        test_cases: List[Dict[str, Any]],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a RAG system against test cases.
        
        Args:
            rag_function: Function that takes query and returns 
                          {"retrieved_ids": [...], "context": "...", "answer": "..."}
            test_cases: List of {"query": "...", "ground_truth_ids": [...]}
            verbose: Print progress
            
        Returns:
            Aggregated metrics and per-query results
        """
        results = []
        all_retrieval_metrics = defaultdict(list)
        all_answer_metrics = defaultdict(list)
        
        for i, case in enumerate(test_cases):
            query = case["query"]
            ground_truth_ids = case.get("ground_truth_ids", [])
            
            if verbose:
                print(f"[{i+1}/{len(test_cases)}] Evaluating: {query[:50]}...")
            
            # Run RAG
            try:
                rag_result = rag_function(query)
                retrieved_ids = rag_result.get("retrieved_ids", [])
                context = rag_result.get("context", "")
                answer = rag_result.get("answer", "")
            except Exception as e:
                logger.error(f"RAG function failed for '{query}': {e}")
                continue
            
            # Evaluate
            eval_result = self.evaluate_single(
                query=query,
                retrieved_ids=retrieved_ids,
                ground_truth_ids=ground_truth_ids,
                context=context,
                answer=answer
            )
            
            results.append(eval_result)
            
            # Collect metrics
            for key, value in eval_result["retrieval"].items():
                all_retrieval_metrics[key].append(value)
            
            if eval_result["answer"]:
                for key, value in eval_result["answer"].items():
                    all_answer_metrics[key].append(value)
        
        # Aggregate
        avg_retrieval = {
            key: sum(values) / len(values)
            for key, values in all_retrieval_metrics.items()
        }
        
        avg_answer = {
            key: sum(values) / len(values)
            for key, values in all_answer_metrics.items()
        } if all_answer_metrics else None
        
        summary = {
            "num_queries": len(test_cases),
            "num_evaluated": len(results),
            "retrieval_metrics": avg_retrieval,
            "answer_metrics": avg_answer,
            "per_query_results": results
        }
        
        if verbose:
            print("\n" + "="*50)
            print("📊 RAG Evaluation Summary")
            print("="*50)
            print(f"\nRetrieval Metrics (averaged over {len(results)} queries):")
            print(f"  Recall@5:    {avg_retrieval.get('recall_at_5', 0):.3f}")
            print(f"  Precision@5: {avg_retrieval.get('precision_at_5', 0):.3f}")
            print(f"  MRR:         {avg_retrieval.get('mrr', 0):.3f}")
            print(f"  NDCG@5:      {avg_retrieval.get('ndcg_at_5', 0):.3f}")
            print(f"  Hit Rate@5:  {avg_retrieval.get('hit_rate_at_5', 0):.3f}")
            
            if avg_answer:
                print(f"\nAnswer Metrics:")
                print(f"  Faithfulness:      {avg_answer.get('faithfulness', 0):.3f}")
                print(f"  Answer Relevance:  {avg_answer.get('answer_relevance', 0):.3f}")
                print(f"  Context Relevance: {avg_answer.get('context_relevance', 0):.3f}")
        
        return summary


# =============================================================================
# TEST DATASET FOR BYNOEMIE
# =============================================================================

BYNOEMIE_TEST_CASES = [
    {
        "query": "romantic dinner dress",
        "ground_truth_ids": ["9773314081058", "9773311525154"],  # Tiara Satin, Monica
    },
    {
        "query": "night out sparkle",
        "ground_truth_ids": ["9763570811170", "9800629813538", "9773308477730"],  # Coco, Sparkle Mini, Maddison
    },
    {
        "query": "main character energy",
        "ground_truth_ids": ["9763570811170", "9773307363618"],  # Coco, Ella
    },
    {
        "query": "boss babe power look",
        "ground_truth_ids": ["9800639152418", "9773298647330", "9773344784674"],  # Kylie Jumpsuit, Alexandria, Zera
    },
    {
        "query": "wedding guest outfit",
        "ground_truth_ids": ["9763569828130", "9773303169314", "9763561242914", "9854144905506"],  # Luna, Camilia, Annabelle, Florina
    },
    {
        "query": "quiet luxury minimalist",
        "ground_truth_ids": ["9769083273506", "9763568451874", "9699997516066"],  # Nana, Sierra, Valeria
    },
    {
        "query": "garden party summer",
        "ground_truth_ids": ["9773303169314", "9854147658018", "9800635351330"],  # Camilia, Dianna, Anna Floral
    },
    {
        "query": "NYE new years eve party",
        "ground_truth_ids": ["9763570811170", "9800629813538", "9773308477730"],  # Coco, Sparkle Mini, Maddison
    },
    {
        "query": "elegant gala red carpet",
        "ground_truth_ids": ["9763569828130", "9763139191074", "9800636465442"],  # Luna, Monica (floor-length), Leslie
    },
    {
        "query": "date night sexy",
        "ground_truth_ids": ["9773314081058", "9763567272226", "9773311525154"],  # Tiara, Leila, Monica
    },
    {
        "query": "beach vacation dress",
        "ground_truth_ids": ["9800635351330", "9800633811234"],  # Anna Floral, Aurel Beach
    },
    {
        "query": "cocktail party dress",
        "ground_truth_ids": ["9763564912930", "9773344784674", "9773344096546"],  # Mimi, Zera, Vela
    },
    {
        "query": "statement clutch bag",
        "ground_truth_ids": ["9773092208930"],  # The Sparkle
    },
    {
        "query": "everyday work bag",
        "ground_truth_ids": ["9775129624866", "9773089653026", "9773087621410"],  # Classic, Harper, Elan
    },
    {
        "query": "fairy tale princess dress",
        "ground_truth_ids": ["9773344096546", "9854144905506"],  # Vela (butterflies), Florina
    },
]


def run_evaluation_demo():
    """Demo evaluation with sample data"""
    print("="*60)
    print("🧪 RAG Evaluation Demo")
    print("="*60)
    
    # Create evaluator
    evaluator = RetrievalEvaluator()
    
    # Sample retrieval results (simulated)
    retrieved_results = [
        ["PROD002", "PROD003", "PROD006", "PROD005", "PROD001"],  # romantic dinner
        ["PROD001", "PROD004", "PROD008", "PROD003", "PROD002"],  # night out sparkle
        ["PROD001", "PROD004", "PROD003", "PROD008", "PROD002"],  # main character
        ["PROD004", "PROD001", "PROD009", "PROD003", "PROD008"],  # boss babe
        ["PROD007", "PROD008", "PROD002", "PROD005", "PROD003"],  # wedding guest
    ]
    
    ground_truth = [
        ["PROD002"],  # romantic dinner → Tiara
        ["PROD001"],  # night out → Coco
        ["PROD001"],  # main character → Coco
        ["PROD004"],  # boss babe → Stella
        ["PROD007", "PROD008"],  # wedding → Bella, Jade
    ]
    
    queries = [
        "romantic dinner dress",
        "night out sparkle",
        "main character energy",
        "boss babe power look",
        "wedding guest outfit"
    ]
    
    # Evaluate
    metrics, results = evaluator.evaluate_batch(
        retrieved_results=retrieved_results,
        ground_truth=ground_truth,
        queries=queries
    )
    
    print("\n📊 Retrieval Evaluation Results")
    print("-"*40)
    print(metrics)
    
    print("\n📝 Per-Query Results:")
    for r in results:
        hit = "✅" if r.retrieval_metrics.get("hit_rate_at_5", 0) > 0 else "❌"
        print(f"  {hit} {r.query[:30]:30} | Recall@5: {r.retrieval_metrics['recall_at_5']:.2f}")
    
    return metrics


if __name__ == "__main__":
    run_evaluation_demo()
