"""
api/server.py
─────────────
Polyglot RAG — FastAPI backend.
Exposes translation-aware query, ingest, and provider-health endpoints.
"""
from __future__ import annotations

import os
import tempfile

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config.settings import settings
from data_wrangling.loader import load_and_split
from data_wrangling.vectorstore import ingest_documents
from search.models.rag_pipeline import run_rag_query
from search.models.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

app = FastAPI(
    title="Polyglot RAG API",
    description=(
        "Translation-aware Retrieval-Augmented Generation backend.\n\n"
        "Dual-model routing: **Gemini** (primary) ↔ **Anthropic Claude** (fallback).\n"
        "Self-querying metadata filters for language-pair, domain, and register."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("═" * 60)
    logger.info("Polyglot RAG API v2.0 — starting up")
    logger.info(f"  Primary provider : {settings.llm_provider.upper()}")
    logger.info(f"  LLM model        : {settings.llm_model}")
    logger.info(f"  Embedding model  : {settings.embedding_model}")
    logger.info(f"  Vector store     : {settings.vectorstore_provider.upper()}")
    gemini_ok = bool(settings.gemini_api_key)
    anthropic_ok = bool(settings.anthropic_api_key)
    logger.info(f"  Gemini key       : {'✓' if gemini_ok else '✗ MISSING'}")
    logger.info(f"  Anthropic key    : {'✓' if anthropic_ok else '✗ MISSING'}")
    logger.info("═" * 60)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
async def health():
    """Liveness check — returns provider availability status."""
    return {
        "status": "ok",
        "primary_provider": settings.llm_provider,
        "model": settings.llm_model,
        "vector_store": settings.vectorstore_provider,
        "providers": {
            "gemini": {
                "available": bool(settings.gemini_api_key),
                "model": settings.llm_model if settings.llm_provider == "gemini" else "gemini-2.0-flash",
            },
            "anthropic": {
                "available": bool(settings.anthropic_api_key),
                "model": settings.llm_model if settings.llm_provider == "anthropic" else "claude-sonnet-4-20250514",
            },
        },
    }


# ── Query (translation-aware RAG) ─────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse, tags=["rag"])
async def query(request: QueryRequest):
    """
    Translate / answer using RAG.

    Pipeline:
    1. **Self-querying** — the LLM extracts a metadata filter (language pair, domain, …)
    2. **Retrieval** — top-k chunks fetched with the filter applied
    3. **Generation** — Gemini primary, Anthropic fallback (or as configured)
    4. **Response** — structured answer with sources and provenance
    """
    try:
        logger.info(f"Query received: {request.query[:80]!r}")
        return run_rag_query(request)
    except ValueError as val_err:
        logger.error(f"Config error: {val_err}")
        raise HTTPException(status_code=400, detail=str(val_err))
    except RuntimeError as rt_err:
        # Both providers failed
        logger.error(f"All providers failed: {rt_err}")
        raise HTTPException(status_code=503, detail=str(rt_err))
    except Exception as exc:
        logger.exception("RAG pipeline error")
        raise HTTPException(status_code=500, detail="Internal processing error") from exc


# ── Ingest ────────────────────────────────────────────────────────────────────

@app.post("/api/ingest", response_model=IngestResponse, tags=["rag"])
async def ingest(request: IngestRequest):
    """
    Ingest a document with rich metadata (language, domain, register).

    Accepts a DocumentMetadata object or a plain dict via the `metadata` field.
    """
    try:
        logger.info(f"Ingest request: {request.source}")

        # Normalise metadata — accept both DocumentMetadata and plain dict
        meta_dict: dict = (
            request.metadata.model_dump()
            if hasattr(request.metadata, "model_dump")
            else dict(request.metadata)
        )

        chunks = load_and_split(
            source=request.source,
            source_type=request.source_type,
            extra_metadata=meta_dict.get("extra", {}),
            source_language=meta_dict.get("source_language"),
            target_language=meta_dict.get("target_language"),
            domain=meta_dict.get("domain", "general"),
            register=meta_dict.get("register", "neutral"),
        )
        added = ingest_documents(chunks)

        return IngestResponse(
            status="ok",
            chunks_added=added,
            source=request.source,
            message=f"Successfully ingested {added} chunks from {request.source}.",
        )
    except Exception as exc:
        logger.exception(f"Ingest failed for {request.source}")
        raise HTTPException(
            status_code=500, detail=f"Ingest failed: {exc}"
        ) from exc


# ── Ingest via file upload (used by the HTML UI) ─────────────────────────────

@app.post("/api/ingest/upload", response_model=IngestResponse, tags=["rag"])
async def ingest_upload(
    file:            UploadFile = File(...),
    source_language: str        = Form(""),
    target_language: str        = Form(""),
    domain:          str        = Form("general"),
    register:        str        = Form("neutral"),
):
    """
    Accept a raw file upload (multipart/form-data) from the HTML UI.
    Saves to a temp file, runs the standard ingest pipeline, then cleans up.
    """
    suffix = os.path.splitext(file.filename)[-1].lower()
    source_type = suffix.lstrip(".")
    if source_type not in {"pdf", "txt", "docx", "csv"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix!r}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        logger.info(f"Upload ingest: {file.filename} ({len(content)} bytes)")

        chunks = load_and_split(
            source=tmp_path,
            source_type=source_type,
            source_language=source_language or None,
            target_language=target_language or None,
            domain=domain,
            register=register,
        )
        # Restore the original filename in metadata
        for chunk in chunks:
            chunk.metadata["source"] = file.filename

        added = ingest_documents(chunks)
        return IngestResponse(
            status="ok",
            chunks_added=added,
            source=file.filename,
            message=f"Successfully ingested {added} chunks from {file.filename}.",
        )
    except Exception as exc:
        logger.exception(f"Upload ingest failed for {file.filename}")
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
