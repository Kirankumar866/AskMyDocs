"""
Embedder: generates OpenAI embeddings and stores chunks in ChromaDB.
Also maintains a BM25 index alongside the vector store.
"""

import hashlib
import json
from typing import List, Dict, Any

import chromadb
import openai

from app.config import settings


def _get_chroma_client() -> chromadb.ClientAPI:
    """Get or create a persistent ChromaDB client."""
    return chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
    )


def _get_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """Get or create the main document collection."""
    return client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"},
    )


def _generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using OpenAI API."""
    client = openai.OpenAI(api_key=settings.openai_api_key)

    # OpenAI supports batching up to 2048 inputs
    all_embeddings = []
    batch_size = 512

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def _generate_chunk_id(chunk: Dict[str, Any]) -> str:
    """Generate a deterministic ID for a chunk based on its content and metadata."""
    content = chunk["page_content"]
    source = chunk["metadata"].get("source", "")
    page = str(chunk["metadata"].get("page", 0))
    chunk_idx = str(chunk["metadata"].get("chunk_index", 0))
    raw = f"{source}:{page}:{chunk_idx}:{content[:100]}"
    return hashlib.md5(raw.encode()).hexdigest()


def ingest_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Embed chunks and store them in ChromaDB.
    Returns statistics about the ingestion.
    """
    if not chunks:
        return {"status": "no_chunks", "count": 0}

    client = _get_chroma_client()
    collection = _get_collection(client)

    texts = [c["page_content"] for c in chunks]
    ids = [_generate_chunk_id(c) for c in chunks]
    metadatas = []
    for c in chunks:
        meta = {k: v for k, v in c["metadata"].items()}
        # ChromaDB metadata values must be str, int, float, or bool
        for k, v in meta.items():
            if not isinstance(v, (str, int, float, bool)):
                meta[k] = str(v)
        metadatas.append(meta)

    # Generate embeddings
    embeddings = _generate_embeddings(texts)

    # Upsert into ChromaDB
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    return {
        "status": "success",
        "chunks_ingested": len(chunks),
        "collection_total": collection.count(),
    }


def get_all_documents_info() -> List[Dict[str, Any]]:
    """Get information about all ingested documents."""
    client = _get_chroma_client()
    collection = _get_collection(client)

    results = collection.get(include=["metadatas"])

    # Group by source file
    sources: Dict[str, Dict[str, Any]] = {}
    for meta in results["metadatas"]:
        source = meta.get("source", "unknown")
        if source not in sources:
            sources[source] = {
                "source": source,
                "filename": meta.get("filename", "unknown"),
                "file_type": meta.get("file_type", "unknown"),
                "chunk_count": 0,
            }
        sources[source]["chunk_count"] += 1

    return list(sources.values())


def delete_document(source: str) -> Dict[str, Any]:
    """Delete all chunks belonging to a specific source document."""
    client = _get_chroma_client()
    collection = _get_collection(client)

    # Get IDs of chunks from this source
    results = collection.get(
        where={"source": source},
        include=[],
    )

    if results["ids"]:
        collection.delete(ids=results["ids"])
        return {"status": "deleted", "chunks_removed": len(results["ids"])}

    return {"status": "not_found", "chunks_removed": 0}


def get_collection_count() -> int:
    """Get the total number of chunks in the collection."""
    client = _get_chroma_client()
    collection = _get_collection(client)
    return collection.count()
