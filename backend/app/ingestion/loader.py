"""
Document loader: reads PDF, Markdown, and plain text files.
Returns a list of LangChain-style Document dicts with page_content + metadata.
"""

import os
from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader


def _load_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from each page of a PDF."""
    reader = PdfReader(file_path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append({
                "page_content": text,
                "metadata": {
                    "source": file_path,
                    "page": i + 1,
                    "file_type": "pdf",
                    "filename": os.path.basename(file_path),
                },
            })
    return docs


def _load_text(file_path: str) -> List[Dict[str, Any]]:
    """Load a plain text or markdown file as a single document."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    ext = Path(file_path).suffix.lower()
    file_type = "markdown" if ext in (".md", ".mdx") else "text"

    if text.strip():
        return [{
            "page_content": text,
            "metadata": {
                "source": file_path,
                "page": 1,
                "file_type": file_type,
                "filename": os.path.basename(file_path),
            },
        }]
    return []


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".mdx", ".text"}


def load_documents(directory: str) -> List[Dict[str, Any]]:
    """
    Recursively load all supported documents from a directory.
    Returns a flat list of document dicts.
    """
    all_docs: List[Dict[str, Any]] = []
    dir_path = Path(directory)

    if not dir_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {directory}")

    for file_path in dir_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            str_path = str(file_path)
            if file_path.suffix.lower() == ".pdf":
                all_docs.extend(_load_pdf(str_path))
            else:
                all_docs.extend(_load_text(str_path))

    return all_docs


def load_single_file(file_path: str) -> List[Dict[str, Any]]:
    """Load a single file and return document dicts."""
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

    if ext == ".pdf":
        return _load_pdf(file_path)
    else:
        return _load_text(file_path)
