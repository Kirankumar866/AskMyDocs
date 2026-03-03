"""
FastAPI Application: Production RAG API — "Ask My Docs"

Endpoints:
  POST /api/ingest          — Upload and ingest documents
  POST /api/ingest/directory — Ingest all documents from the docs/ folder
  POST /api/query           — Ask a question (full RAG pipeline)
  GET  /api/documents       — List all ingested documents
  DELETE /api/documents     — Delete a document by source path
  GET  /api/health          — Health check
  POST /api/evaluate        — Run CI evaluation pipeline
"""

import os
import shutil
import tempfile
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.ingestion.loader import load_documents, load_single_file, SUPPORTED_EXTENSIONS
from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import ingest_chunks, get_all_documents_info, delete_document, get_collection_count
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.reranker import rerank
from app.generation.generator import generate_answer


# ── App Setup ─────────────────────────────────────────────────────────

app = FastAPI(
    title="Ask My Docs — Production RAG API",
    description="A domain-specific RAG system with hybrid retrieval, cross-encoder reranking, and citation enforcement.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    citations: list
    confidence: str
    sources: list
    model: str
    chunks_used: int


class DeleteRequest(BaseModel):
    source: str


class IngestResponse(BaseModel):
    status: str
    documents_loaded: int = 0
    chunks_created: int = 0
    chunks_ingested: int = 0
    collection_total: int = 0


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "collection_size": get_collection_count(),
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
    }


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_uploaded_files(files: List[UploadFile] = File(...)):
    """Upload and ingest one or more document files."""
    all_docs = []

    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}",
            )

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            docs = load_single_file(tmp_path)
            # Replace temp path with original filename in metadata
            for doc in docs:
                doc["metadata"]["source"] = file.filename
                doc["metadata"]["filename"] = file.filename
            all_docs.extend(docs)
        finally:
            os.unlink(tmp_path)

    if not all_docs:
        return IngestResponse(status="no_content", documents_loaded=0)

    chunks = chunk_documents(all_docs)
    result = ingest_chunks(chunks)

    return IngestResponse(
        status=result["status"],
        documents_loaded=len(all_docs),
        chunks_created=len(chunks),
        chunks_ingested=result["chunks_ingested"],
        collection_total=result["collection_total"],
    )


@app.post("/api/ingest/directory", response_model=IngestResponse)
async def ingest_from_directory():
    """Ingest all documents from the configured docs/ directory."""
    docs_path = Path(settings.docs_dir)
    if not docs_path.exists():
        docs_path.mkdir(parents=True, exist_ok=True)
        return IngestResponse(
            status="empty_directory",
            documents_loaded=0,
        )

    docs = load_documents(settings.docs_dir)
    if not docs:
        return IngestResponse(status="no_documents_found", documents_loaded=0)

    chunks = chunk_documents(docs)
    result = ingest_chunks(chunks)

    return IngestResponse(
        status=result["status"],
        documents_loaded=len(docs),
        chunks_created=len(chunks),
        chunks_ingested=result["chunks_ingested"],
        collection_total=result["collection_total"],
    )


@app.post("/api/query")
async def query_documents(request: QueryRequest):
    """
    Full RAG pipeline:
    1. Hybrid search (BM25 + vector)
    2. Cross-encoder reranking
    3. LLM generation with citation enforcement
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Step 1: Hybrid retrieval
    candidates = hybrid_search(request.question)

    if not candidates:
        return {
            "answer": "No documents found. Please upload some documents first.",
            "citations": [],
            "confidence": "low",
            "sources": [],
            "model": settings.llm_model,
            "chunks_used": 0,
        }

    # Step 2: Cross-encoder reranking
    reranked = rerank(request.question, candidates, top_k=request.top_k)

    # Step 3: LLM generation with citations
    response = generate_answer(request.question, reranked)

    return response


@app.get("/api/documents")
async def list_documents():
    """List all ingested documents and their chunk counts."""
    docs = get_all_documents_info()
    return {
        "documents": docs,
        "total_documents": len(docs),
        "total_chunks": get_collection_count(),
    }


@app.delete("/api/documents")
async def remove_document(request: DeleteRequest):
    """Delete all chunks belonging to a specific source document."""
    result = delete_document(request.source)
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Document not found")
    return result


@app.post("/api/evaluate")
async def run_evaluation():
    """Run the CI evaluation pipeline against the golden dataset."""
    from app.evaluation.evaluate import run_evaluation as eval_fn
    result = eval_fn()
    return result
