# AskMyDocs
Production RAG Application: "Ask My Docs"

This document outlines the architecture and implementation plan for a domain-specific RAG system featuring hybrid retrieval, cross-encoder reranking, citation enforcement, and CI-gated evaluation.

## Frontend Development (Next.js)

First, run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Backend Development (FastAPI)

```bash
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
