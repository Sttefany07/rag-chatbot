from __future__ import annotations
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

"""
ee la configuración del proyecto desde las variables de entorno.
"""
load_dotenv()

def _get_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

class Settings(BaseModel):
    # LLM
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    # Embeddings
    embeddings_provider: str = os.getenv("EMBEDDINGS_PROVIDER", "ollama")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")

    # Chroma / RAG
    chroma_path: str = os.getenv("CHROMA_PATH", "./data/chroma")
    max_context_chunks: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "6"))
    mmr: bool = _get_bool("MMR", True)
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "220"))
    ingest_batch_size: int = int(os.getenv("INGEST_BATCH_SIZE", "256"))

    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])

settings = Settings()
