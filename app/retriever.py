from __future__ import annotations
from typing import List, Dict, Any, Iterable
import chromadb

from app.settings import settings
from app.embeddings import get_embeddings

"""
Busca en la base vectorial Chroma los documentos más parecidos a una consulta.
"""

__all__ = ["get_collection", "search", "build_embedding_function"]

class _LocalEmbeddingFunction:

    def __init__(self):
        self._embedder = get_embeddings()

    def __call__(self, input: Iterable[str]) -> List[List[float]]:
        texts: List[str] = list(input)
        return self._embedder.embed(texts)


# Función que construye la función de embeddings según el proveedor

def build_embedding_function(provider: str):
    prov = (provider or "").strip().lower()
    if prov == "ollama":
        return _LocalEmbeddingFunction()
    raise NotImplementedError(f"Embeddings provider '{provider}' no implementado")

def get_collection():
    client = chromadb.PersistentClient(path=settings.chroma_path)
    ef = build_embedding_function(settings.embeddings_provider)
    return client.get_or_create_collection(
        name="docs",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

def search(query: str, k: int = 6) -> List[Dict[str, Any]]:
    col = get_collection()
    res = col.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    ids = (res.get("ids") or [[]])[0] if res.get("ids") else [f"doc:{i}" for i in range(len(docs))]

    out: List[Dict[str, Any]] = []
    # unimos cada documento con sus metadatos
    for t, m, d, i in zip(docs, metas, dists, ids):
        out.append({
            "id": i,
            "text": t,
            "source": (m or {}).get("source"),
            "page": (m or {}).get("page"),
            "distance": d,
        })
    return out
