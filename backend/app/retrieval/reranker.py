"""
Cross-Encoder Reranker: takes the top candidates from hybrid search
and re-scores them using a cross-encoder model for much higher precision.
"""

from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

from app.config import settings

# Lazy-loaded singleton
_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    """Lazy-load the cross-encoder model."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.cross_encoder_model)
    return _reranker


def rerank(
    query: str,
    documents: List[Dict[str, Any]],
    top_k: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Rerank documents using a cross-encoder model.
    
    The cross-encoder scores each (query, document) pair independently,
    producing much more accurate relevance scores than bi-encoder similarity.

    Args:
        query: The user's question
        documents: List of candidate documents from hybrid search
        top_k: Number of top results to return after reranking
    
    Returns:
        Reranked list of documents with cross-encoder scores
    """
    if not documents:
        return []

    _top_k = top_k or settings.rerank_top_k

    reranker = _get_reranker()

    # Create (query, doc) pairs for the cross-encoder
    pairs = [(query, doc["content"]) for doc in documents]

    # Score all pairs
    scores = reranker.predict(pairs)

    # Attach scores and sort
    for i, doc in enumerate(documents):
        doc["rerank_score"] = float(scores[i])

    # Sort by rerank score descending
    reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

    return reranked[:_top_k]
