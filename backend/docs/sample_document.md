# Ask My Docs — Sample Document

## About This Project

This is a production-ready Retrieval-Augmented Generation (RAG) application called "Ask My Docs."
It allows users to upload their own documents and ask natural language questions about them.

## Key Features

### Hybrid Retrieval
The system uses a combination of dense vector search (via OpenAI embeddings + ChromaDB) and sparse keyword search (BM25) to retrieve the most relevant document chunks. These results are fused using Reciprocal Rank Fusion (RRF).

### Cross-Encoder Reranking
After initial retrieval, a cross-encoder model re-scores each candidate to improve precision. This is significantly more accurate than bi-encoder similarity alone.

### Citation Enforcement
The LLM is instructed to cite specific document chunks in its answers. Each citation maps back to the original source document and page number.

### CI-Gated Evaluation
An automated evaluation pipeline can be run on every pull request. It tests the RAG system against a golden dataset and blocks merges if faithfulness, relevancy, or precision scores fall below configurable thresholds.

## Architecture

1. **Ingestion Pipeline**: Documents → Loader → Chunker → Embedder → ChromaDB
2. **Retrieval Pipeline**: Query → Hybrid Search (Vector + BM25) → RRF → Reranker
3. **Generation Pipeline**: Reranked Chunks → Prompt with Citations → OpenAI GPT-4o → Structured JSON

## Supported File Types

- PDF (.pdf)
- Markdown (.md, .mdx)
- Plain Text (.txt)

## Getting Started

1. Copy `.env.example` to `.env` and add your OpenAI API key
2. Install dependencies: `pip install -r requirements.txt`
3. Place your documents in the `docs/` folder
4. Run: `uvicorn app.main:app --reload`
5. Visit the API docs at http://localhost:8000/docs
