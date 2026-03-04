"""
api/server.py
Updated for 2026: Added Gemini support and enhanced error reporting.
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config.settings import settings
from data_wrangling.loader import load_and_split
from data_wrangling.vectorstore import ingest_documents
from search.models.rag_pipeline import run_rag_query
from search.models.schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse

app = FastAPI(
    title="Polyglot RAG API",
    description="Retrieval-Augmented Generation backend — Gemini / Anthropic / OpenAI + Chroma / Pinecone",
    version="1.1.0",
)

# Standard CORS setup to allow your Streamlit or other front-ends to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup Events ────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Polyglot RAG API...")
    logger.info(f"Provider: {settings.llm_provider.upper()} | Model: {settings.llm_model}")
    logger.info(f"Vector Store: {settings.vectorstore_provider.upper()}")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
async def health():
    """Liveness check to verify API and provider status."""
    return {
        "status": "ok", 
        "provider": settings.llm_provider, 
        "model": settings.llm_model,
        "vector_store": settings.vectorstore_provider
    }


# ── Query ─────────────────────────────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse, tags=["rag"])
async def query(request: QueryRequest):
    """
    Ask a question. The pipeline:
    1. Embeds the query
    2. Retrieves top-k chunks from the vector store
    3. Sends context + question to the LLM (Gemini/Anthropic)
    4. Returns a structured answer with sources
    """
    try:
        logger.info(f"Received query: {request.query[:50]}...")
        return run_rag_query(request)
    except ValueError as val_err:
        # Catch provider/configuration errors
        logger.error(f"Configuration error: {val_err}")
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        logger.exception("RAG Pipeline failed")
        raise HTTPException(status_code=500, detail="Internal processing error") from exc


# ── Ingest ────────────────────────────────────────────────────────────────────

@app.post("/api/ingest", response_model=IngestResponse, tags=["rag"])
async def ingest(request: IngestRequest):
    """
    Load a document (file path or URL), chunk it, embed it, and store it.
    """
    try:
        logger.info(f"Ingesting source: {request.source}")
        chunks = load_and_split(
            source=request.source,
            source_type=request.source_type,
            extra_metadata=request.metadata,
        )
        added = ingest_documents(chunks)
        
        return IngestResponse(
            status="ok", 
            chunks_added=added, 
            source=request.source,
            message=f"Successfully ingested {added} chunks."
        )
    except Exception as exc:
        logger.exception(f"Ingest failed for {request.source}")
        raise HTTPException(status_code=500, detail=f"Ingest failed: {str(exc)}") from exc


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Ensure logs directory exists if logging to file
    uvicorn.run(
        "api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )