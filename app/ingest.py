from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import re
import unicodedata

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.settings import settings
from app.retriever import get_collection

"""
ingesta de documentos PDF en el sistema RAG.
"""

def _hash_bytes(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()[:16]

# Limpia el texto para evitar caracteres raros
def _sanitize_text(s: str) -> str:
    if not s:
        return ""
    # Normaliza caracteres a forma estándar
    s = unicodedata.normalize("NFKC", s)
    # Elimina surrogates no emparejados y NULs
    s = re.sub(r"[\ud800-\udfff]", "", s)
    s = s.replace("\x00", "")
    # Quita la mayoría de control chars pero mantiene saltos y tabs
    s = "".join(c for c in s if c.isprintable() or c in "\n\t ")
    return s.strip()

# Lee un PDF y devuelve lista de páginas con su número y texto
def load_pdf_texts(pdf_path: Path) -> List[dict]:
    reader = PdfReader(str(pdf_path))
    pages: List[dict] = []
    for i, page in enumerate(reader.pages, start=1):
        # texto crudo de la página
        raw = page.extract_text() or ""
        # limpiamos el texto
        text = _sanitize_text(raw)
        pages.append({"page": i, "text": text})
    return pages


def chunk_pages(pages: List[dict]) -> List[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", ".", " "],
    )
    chunks: List[dict] = []
    for p in pages:
        if not p["text"]:
            continue
        parts = splitter.split_text(p["text"])
        for idx, c in enumerate(parts):
            c = _sanitize_text(c)
            if c:
                chunks.append({"page": p["page"], "text": c, "chunk": idx})
    return chunks


def ingest_pdf(pdf_path: Path, source_name: str | None = None) -> Dict[str, Any]:
    source_name = source_name or pdf_path.name
    raw = pdf_path.read_bytes()
    file_hash = _hash_bytes(raw)

    pages = load_pdf_texts(pdf_path)
    chunks = chunk_pages(pages)

    col = get_collection()

    if not chunks:
        # si no hay texto, puede ser PDF escaneado sin OCR
        return {
            "file_hash": file_hash,
            "source": source_name,
            "added_chunks": 0,
            "collection_count": col.count(),
            "note": "El PDF no tiene texto extraíble (¿escaneado sin OCR?) o todo quedó vacío tras limpieza.",
        }

    # Borra lo anterior de esta misma fuente para evitar duplicados
    col.delete(where={"source": {"$eq": source_name}})

    ids = [f"{file_hash}:{c['page']}:{c['chunk']}" for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [
        {"source": source_name, "page": c["page"], "chunk": c["chunk"], "file_hash": file_hash}
        for c in chunks
    ]

    # Guardamos en la colección
    col.add(ids=ids, documents=texts, metadatas=metadatas)

    return {
        "file_hash": file_hash,
        "source": source_name,
        "added_chunks": len(texts),
        "collection_count": col.count(),
    }
