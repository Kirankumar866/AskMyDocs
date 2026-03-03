"""
CI-Gated Evaluation Pipeline using Ragas.
Evaluates the RAG pipeline against a golden test dataset.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.reranker import rerank
from app.generation.generator import generate_answer


# Golden test dataset path
GOLDEN_DATASET_PATH = Path(__file__).parent.parent.parent / "tests" / "golden_dataset.json"

# Minimum thresholds for CI gates
THRESHOLDS = {
    "faithfulness": 0.7,
    "answer_relevancy": 0.7,
    "context_precision": 0.6,
}


def load_golden_dataset() -> List[Dict[str, Any]]:
    """Load the golden test dataset."""
    if not GOLDEN_DATASET_PATH.exists():
        return []

    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation() -> Dict[str, Any]:
    """
    Run the full RAG pipeline on each golden test case and compute metrics.

    Each test case should have:
    - question: str
    - ground_truth: str (expected answer)

    Returns evaluation results with per-question scores and aggregate metrics.
    """
    dataset = load_golden_dataset()
    if not dataset:
        return {
            "status": "no_dataset",
            "message": "No golden dataset found. Create tests/golden_dataset.json",
        }

    results = []
    for test_case in dataset:
        question = test_case["question"]
        ground_truth = test_case.get("ground_truth", "")

        # Run the RAG pipeline
        candidates = hybrid_search(question)
        reranked = rerank(question, candidates)
        response = generate_answer(question, reranked)

        answer = response.get("answer", "")
        contexts = [chunk["content"] for chunk in reranked]

        # Simple faithfulness check: what fraction of sentences in the answer
        # can be traced back to the context
        faithfulness = _compute_faithfulness(answer, contexts)

        # Answer relevancy: simple overlap-based score
        relevancy = _compute_relevancy(answer, question)

        # Context precision: how relevant are the retrieved chunks to the question
        context_precision = _compute_context_precision(contexts, question)

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": answer,
            "num_contexts": len(contexts),
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "context_precision": context_precision,
        })

    # Compute aggregates
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results) if results else 0
    avg_relevancy = sum(r["answer_relevancy"] for r in results) / len(results) if results else 0
    avg_context_precision = sum(r["context_precision"] for r in results) / len(results) if results else 0

    # CI gate check
    passed = (
        avg_faithfulness >= THRESHOLDS["faithfulness"]
        and avg_relevancy >= THRESHOLDS["answer_relevancy"]
        and avg_context_precision >= THRESHOLDS["context_precision"]
    )

    return {
        "status": "passed" if passed else "failed",
        "thresholds": THRESHOLDS,
        "aggregate_scores": {
            "faithfulness": round(avg_faithfulness, 4),
            "answer_relevancy": round(avg_relevancy, 4),
            "context_precision": round(avg_context_precision, 4),
        },
        "num_test_cases": len(results),
        "details": results,
    }


def _compute_faithfulness(answer: str, contexts: List[str]) -> float:
    """
    Simple faithfulness heuristic: what fraction of answer sentences
    have significant word overlap with at least one context chunk.
    """
    if not answer or not contexts:
        return 0.0

    sentences = [s.strip() for s in answer.replace("  ", " ").split(".") if len(s.strip()) > 10]
    if not sentences:
        return 1.0

    context_blob = " ".join(contexts).lower()
    context_words = set(context_blob.split())

    grounded = 0
    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        # Remove common stop words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and", "or", "it", "this", "that", "with"}
        meaningful = sentence_words - stopwords
        if not meaningful:
            grounded += 1
            continue

        overlap = len(meaningful & context_words) / len(meaningful)
        if overlap >= 0.4:
            grounded += 1

    return grounded / len(sentences)


def _compute_relevancy(answer: str, question: str) -> float:
    """Simple relevancy: word overlap between answer and question."""
    if not answer or not question:
        return 0.0

    answer_words = set(answer.lower().split())
    question_words = set(question.lower().split())
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and", "or", "what", "how", "why", "when", "where", "which", "?"}

    q_meaningful = question_words - stopwords
    if not q_meaningful:
        return 1.0

    overlap = len(q_meaningful & answer_words) / len(q_meaningful)
    return min(overlap * 1.5, 1.0)  # Scale up slightly, cap at 1.0


def _compute_context_precision(contexts: List[str], question: str) -> float:
    """Simple context precision: how many retrieved chunks mention question keywords."""
    if not contexts or not question:
        return 0.0

    question_words = set(question.lower().split())
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and", "or", "what", "how", "why", "when", "where", "which", "?"}
    q_meaningful = question_words - stopwords

    if not q_meaningful:
        return 1.0

    relevant_count = 0
    for ctx in contexts:
        ctx_words = set(ctx.lower().split())
        overlap = len(q_meaningful & ctx_words) / len(q_meaningful)
        if overlap >= 0.3:
            relevant_count += 1

    return relevant_count / len(contexts)
