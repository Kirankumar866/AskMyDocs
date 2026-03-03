"""
Document chunker: splits documents into smaller, overlapping chunks
suitable for embedding and retrieval.
"""

from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    Each chunk inherits the metadata of its parent document plus a chunk_index.
    """
    _chunk_size = chunk_size or settings.chunk_size
    _chunk_overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_chunk_size,
        chunk_overlap=_chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunked_docs: List[Dict[str, Any]] = []

    for doc in documents:
        text = doc["page_content"]
        metadata = doc["metadata"]

        splits = splitter.split_text(text)
        for i, chunk_text in enumerate(splits):
            chunked_docs.append({
                "page_content": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(splits),
                },
            })

    return chunked_docs
