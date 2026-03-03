"""
Application configuration loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o"

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma"

    # Documents
    docs_dir: str = "./docs"

    # Cross-Encoder Reranker
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval Parameters
    bm25_top_k: int = 20
    vector_top_k: int = 20
    rerank_top_k: int = 5

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
