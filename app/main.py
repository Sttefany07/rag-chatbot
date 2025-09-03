from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.settings import settings
from app.logging_config import configure_logging
from app.schemas import ChatRequest, ChatResponse, SourceChunk
from app.ingest import ingest_pdf
from app.rag import retrieve_context, build_messages
from app.llm import get_llm
from app.embeddings import get_embeddings

logger = configure_logging()
app = FastAPI(title="RAG Chatbot FastAPI", version="1.0.0")

# Recuerda el último archivo subido para filtrar el contexto por defecto
LAST_SOURCE: Optional[str] = None

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    html_path = Path(__file__).parent / "ui" / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>RAG Chatbot</h1><p>Frontend no encontrado. Usa /docs</p>")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Sube un PDF, lo trocea e ingesta en Chroma. Guarda el nombre como LAST_SOURCE.
    """
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Solo se aceptan PDFs")

        uploads_dir = Path("data/uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)

        dest = uploads_dir / file.filename
        dest.write_bytes(await file.read())

        res = ingest_pdf(dest, source_name=file.filename)
        logger.info("ingest.ok", extra={"file": file.filename, **res})

        # recuerda la fuente actual para el chat
        global LAST_SOURCE
        LAST_SOURCE = file.filename

        # devolvemos también la fuente actual
        return {**res, "source": file.filename}
    except Exception as e:
        logger.exception("ingest.error", extra={"err": str(e)})
        raise HTTPException(status_code=500, detail=f"Ingesta fallida: {e}")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):

    try:
        top_k = req.top_k or settings.max_context_chunks

        # si tu ChatRequest no tiene 'source', getattr devolverá None
        requested_source: Optional[str] = getattr(req, "source", None)
        current_source: Optional[str] = requested_source or LAST_SOURCE

        ctx = retrieve_context(req.message, top_k=top_k, source=current_source)

        contexts: List[str] = ctx["contexts"]
        metas = ctx["metas"]
        ids = ctx["ids"]
        dists = ctx["distances"]

        if not contexts:
            return ChatResponse(
                answer="No encontré información relevante en la base de conocimientos para responder.",
                used_sources=[],
                model=None,
                extra={"top_k": top_k, "mmr": getattr(req, "mmr", None), "source": current_source},
            )

        # Construir mensajes e invocar LLM
        history = [m.model_dump() for m in (req.history or [])]
        msgs = build_messages(req.message, contexts, metas, history)

        llm = get_llm()
        out = llm.chat(msgs)
        answer = out.get("content", "")

        used = [
            SourceChunk(
                id=ids[i],
                source=str((metas[i] or {}).get("source", "desconocido")),
                page=(metas[i] or {}).get("page"),
                distance=float(dists[i]) if dists and i < len(dists) else None,
                text=contexts[i][:5000],
            )
            for i in range(len(contexts))
        ]

        resp = ChatResponse(
            answer=answer,
            used_sources=used,
            model=out.get("model"),
            extra={
                "top_k": top_k,
                "mmr": getattr(req, "mmr", None) if getattr(req, "mmr", None) is not None else getattr(settings, "mmr", None),
                "source": current_source,
            },
        )
        logger.info("chat.ok", extra={"top_k": top_k, "n_ctx": len(contexts), "source": current_source})
        return resp
    except Exception as e:
        logger.exception("chat.error", extra={"err": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/embed")
def debug_embed(q: str = Query("hola mundo")):

    try:
        emb = get_embeddings()
        vecs = emb.embed([q])
        return JSONResponse({"len": len(vecs[0]), "preview": vecs[0][:8]})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
