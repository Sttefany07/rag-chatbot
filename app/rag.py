from __future__ import annotations
from typing import Any, Dict, List, Optional

from app.retriever import get_collection

# Función para recuperar contexto desde la base vectorial (Chroma)

def retrieve_context(question: str, top_k: int, source: Optional[str] = None) -> Dict[str, Any]:
    """
    busca en Chroma los fragmentos más parecidos a la pregunta.
    """
    col = get_collection()
    n_initial = max(top_k * 3, top_k)

    query_kwargs: Dict[str, Any] = dict(
        query_texts=[question],
        n_results=n_initial,
        include=["documents", "metadatas", "distances"],  # en Chroma 0.5.x no existe "ids" en include
    )
    if source:
        query_kwargs["where"] = {"source": {"$eq": source}}

    res = col.query(**query_kwargs)

    docs: List[str] = (res.get("documents") or [[]])[0]
    metas: List[dict] = (res.get("metadatas") or [[]])[0]
    dists: List[float] = (res.get("distances") or [[]])[0]

    if not docs:
        return {"contexts": [], "ids": [], "metas": [], "distances": []}

    # Construye IDs a partir de metadatos si están, si no, enumera.
    ids: List[str] = []
    if metas:
        for m in metas:
            if isinstance(m, dict) and ("file_hash" in m or "page" in m or "chunk" in m):
                ids.append(f"{m.get('file_hash', 'doc')}:{m.get('page', '?')}:{m.get('chunk', '?')}")
            else:
                ids.append("doc")
    else:
        ids = [f"doc:{j}" for j in range(len(docs))]

    # Orden por menor distancia (cosine): menor = más parecido
    if dists:
        order = sorted(range(len(docs)), key=lambda j: (dists[j] if dists[j] is not None else 1e9))
    else:
        order = list(range(len(docs)))

    idxs = order[: min(top_k, len(order))]

    return {
        "contexts": [docs[i] for i in idxs],
        "ids": [ids[i] for i in idxs],
        "metas": [metas[i] for i in idxs],
        "distances": [dists[i] for i in idxs] if dists else [],
    }

"""
Arma el prompt para el LLM con instrucciones, historial y contexto
"""
def build_messages(
    question: str,
    contexts: List[str],
    metas: List[dict],
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:

    """
    Construye el prompt para el LLM con:
    - instrucciones de sistema
    - historial (si existe)
    - la pregunta del usuario + fragmentos de contexto
    """
    history = history or []

    def fmt_meta(m: dict | None) -> str:
        m = m or {}
        src = m.get("source", "desconocido")
        page = m.get("page")
        if page is not None:
            return f"{src} (p. {page})"
        return str(src)

    blocks: List[str] = []
    for i, ctx in enumerate(contexts):
        meta_str = fmt_meta(metas[i] if metas and i < len(metas) else {})
        blocks.append(f"[{i+1}] {meta_str}\n{ctx}".strip())
    context_block = "\n\n".join(blocks) if blocks else "N/A"

    # Mensaje de sistema con instrucciones claras
    system_msg = (
        "Eres un asistente útil. Responde SIEMPRE en español.\n"
        "Usa exclusivamente la información en el CONTEXTO para responder.\n"
        "Si no hay suficiente información en el contexto, dilo claramente."
    )

    # Mensaje de usuario con la pregunta y el contexto numerado
    user_msg = (
        f"Pregunta: {question}\n\n"
        f"CONTEXTO (fragmentos numerados):\n{context_block}\n\n"
        "Cuando cites, referencia los fragmentos así: [1], [2]."
    )

    # Lista final de mensajes para el LLM
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_msg}]

    # Agregamos el historial si existe (conversaciones previas)
    for m in history:
        r = m.get("role")
        c = m.get("content")
        if r in {"user", "assistant", "system"} and isinstance(c, str):
            messages.append({"role": r, "content": c})

    # Finalmente añadimos la nueva pregunta del usuario
    messages.append({"role": "user", "content": user_msg})
    return messages
