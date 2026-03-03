"""
models/schemas.py
─────────────────
Pydantic v2 data models shared across the entire application.
"""

from __future__ import annotations

from datetime import datetime, timezone  # Updated import
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Inbound API ───────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Payload the client sends to /api/query."""
    query: str = Field(..., min_length=1, max_length=4096, description="User question")
    top_k: int = Field(5, ge=1, le=20, description="Number of retrieved chunks")
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    metadata_filter: dict[str, Any] | None = None


class IngestRequest(BaseModel):
    """Payload for ingesting a file path or URL."""
    source: str = Field(..., description="Absolute path or public URL")
    source_type: str = Field("pdf", description="pdf | txt | docx | url | csv")
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Retrieved chunk ───────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    source: str
    page: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Outbound API ──────────────────────────────────────────────────────────────

class QueryResponse(BaseModel):
    """Full answer envelope returned to the client."""
    session_id: str
    query: str
    answer: str
    sources: list[RetrievedChunk]
    model: str
    latency_ms: float
    # FIX: Use timezone-aware datetime for 2026 standards
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class IngestResponse(BaseModel):
    status: str               # "ok" | "error"
    chunks_added: int
    source: str
    message: str = ""