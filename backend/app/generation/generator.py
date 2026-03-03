"""
LLM Generator with Citation Enforcement.
Generates answers from context chunks and forces the model to cite sources.
"""

import json
from typing import List, Dict, Any

import openai

from app.config import settings


SYSTEM_PROMPT = """You are a precise, helpful document assistant for an "Ask My Docs" system.
Your job is to answer questions ONLY based on the provided context chunks.

## RULES:
1. Answer the question using ONLY the information in the provided context chunks.
2. You MUST cite your sources using bracket notation like [1], [2], etc.
3. Each citation number corresponds to a context chunk number.
4. If the context does not contain enough information to answer, say:
   "I don't have enough information in the provided documents to answer this question."
5. Do NOT make up information or use knowledge outside the provided context.
6. Be concise but thorough. Use bullet points or numbered lists when appropriate.
7. If multiple chunks support a claim, cite all of them, e.g., [1][3].

## RESPONSE FORMAT:
Return your response as valid JSON with this exact structure:
{
  "answer": "Your detailed answer with inline citations like [1] and [2]...",
  "citations": [
    {
      "chunk_id": 1,
      "source": "filename or source path",
      "relevance": "Brief explanation of why this chunk was cited"
    }
  ],
  "confidence": "high" | "medium" | "low"
}
"""


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format reranked chunks into a numbered context block for the prompt."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("filename", chunk["metadata"].get("source", "unknown"))
        page = chunk["metadata"].get("page", "N/A")
        context_parts.append(
            f"--- CHUNK [{i}] ---\n"
            f"Source: {source} | Page: {page}\n"
            f"Content:\n{chunk['content']}\n"
        )
    return "\n".join(context_parts)


def generate_answer(
    query: str,
    context_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate an answer with enforced citations.

    Args:
        query: The user's question
        context_chunks: Reranked context chunks from the retrieval pipeline

    Returns:
        Dict with 'answer', 'citations', 'confidence', and 'sources' keys
    """
    if not context_chunks:
        return {
            "answer": "I don't have any documents to search through. Please upload some documents first.",
            "citations": [],
            "confidence": "low",
            "sources": [],
        }

    context_str = _format_context(context_chunks)

    user_message = f"""## CONTEXT CHUNKS:
{context_str}

## USER QUESTION:
{query}

Remember: Return your response as valid JSON with "answer", "citations", and "confidence" fields.
Cite sources using [1], [2], etc. corresponding to the chunk numbers above."""

    client = openai.OpenAI(api_key=settings.openai_api_key)

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    raw_content = response.choices[0].message.content

    # Parse the structured response
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        parsed = {
            "answer": raw_content,
            "citations": [],
            "confidence": "low",
        }

    # Enrich citations with full source metadata
    sources = []
    for i, chunk in enumerate(context_chunks, 1):
        sources.append({
            "chunk_number": i,
            "source": chunk["metadata"].get("source", "unknown"),
            "filename": chunk["metadata"].get("filename", "unknown"),
            "page": chunk["metadata"].get("page", "N/A"),
            "content_preview": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
            "rerank_score": chunk.get("rerank_score", None),
        })

    parsed["sources"] = sources
    parsed["model"] = settings.llm_model
    parsed["chunks_used"] = len(context_chunks)

    return parsed
