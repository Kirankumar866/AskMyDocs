"""
Hybrid Search: combines dense vector search (ChromaDB) with sparse BM25 search
using Reciprocal Rank Fusion (RRF) to produce a unified ranking.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

import chromadb
from rank_bm25 import BM25Okapi
import openai

from app.config import settings


def _get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
    )


def _get_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    return client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"},
    )


def _embed_query(query: str) -> List[float]:
    """Generate embedding for the query using OpenAI."""
    client = openai.OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=query,
    )
    return response.data[0].embedding


def _vector_search(
    collection: chromadb.Collection,
    query_embedding: List[float],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Dense vector similarity search via ChromaDB."""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    for i in range(len(results["ids"][0])):
        docs.append({
            "id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "vector_score": 1 - results["distances"][0][i],  # cosine distance -> similarity
        })
    return docs


def _bm25_search(
    query: str,
    all_docs: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Sparse BM25 keyword search over all stored documents."""
    if not all_docs:
        return []

    # Tokenize documents
    tokenized_corpus = [doc["content"].lower().split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)

    # Score query
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)

    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # Only include docs with positive BM25 score
            doc = all_docs[idx].copy()
            doc["bm25_score"] = float(scores[idx])
            results.append(doc)

    return results


def _reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Combine vector and BM25 results using Reciprocal Rank Fusion.
    RRF score = sum( 1 / (k + rank) ) across all ranking lists.
    """
    # Build a map of doc_id -> fused score + doc data
    fused: Dict[str, Dict[str, Any]] = {}

    for rank, doc in enumerate(vector_results):
        doc_id = doc["id"]
        rrf_score = 1.0 / (k + rank + 1)
        if doc_id not in fused:
            fused[doc_id] = {
                "id": doc_id,
                "content": doc["content"],
                "metadata": doc["metadata"],
                "rrf_score": 0.0,
                "vector_rank": rank + 1,
                "bm25_rank": None,
            }
        fused[doc_id]["rrf_score"] += rrf_score
        fused[doc_id]["vector_score"] = doc.get("vector_score", 0)

    for rank, doc in enumerate(bm25_results):
        doc_id = doc["id"]
        rrf_score = 1.0 / (k + rank + 1)
        if doc_id not in fused:
            fused[doc_id] = {
                "id": doc_id,
                "content": doc["content"],
                "metadata": doc["metadata"],
                "rrf_score": 0.0,
                "vector_rank": None,
                "bm25_rank": rank + 1,
            }
        fused[doc_id]["rrf_score"] += rrf_score
        fused[doc_id]["bm25_rank"] = rank + 1
        fused[doc_id]["bm25_score"] = doc.get("bm25_score", 0)

    # Sort by fused RRF score descending
    sorted_results = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
    return sorted_results


def hybrid_search(query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
    """
    Perform hybrid search: dense vector + BM25 sparse, fused with RRF.
    Returns a ranked list of document chunks.
    """
    client = _get_chroma_client()
    collection = _get_collection(client)

    total_docs = collection.count()
    if total_docs == 0:
        return []

    # 1. Dense vector search
    query_embedding = _embed_query(query)
    vector_results = _vector_search(
        collection,
        query_embedding,
        min(settings.vector_top_k, total_docs),
    )

    # 2. BM25 sparse search — fetch all documents for BM25 scoring
    all_stored = collection.get(include=["documents", "metadatas"])
    all_docs_for_bm25 = []
    for i in range(len(all_stored["ids"])):
        all_docs_for_bm25.append({
            "id": all_stored["ids"][i],
            "content": all_stored["documents"][i],
            "metadata": all_stored["metadatas"][i],
        })

    bm25_results = _bm25_search(
        query,
        all_docs_for_bm25,
        min(settings.bm25_top_k, total_docs),
    )

    # 3. Reciprocal Rank Fusion
    fused_results = _reciprocal_rank_fusion(vector_results, bm25_results)

    # Return top-k fused results
    _top_k = top_k or (settings.vector_top_k + settings.bm25_top_k)
    return fused_results[:_top_k]
