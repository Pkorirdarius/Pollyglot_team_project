"""
search/models/schemas.py
─────────────────────────
Pydantic v2 data models shared across the entire application.
Expanded with self-querying metadata schema for language-aware RAG filtering.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Self-Querying Metadata Schema ─────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    """
    Rich metadata attached to every ingested chunk.
    Enables self-querying retrieval: the LLM can generate structured filters
    (e.g. source_language="fr", domain="legal") before hitting the vector store.
    """
    # Identity
    source: str = Field(..., description="Filename or URL of the original document")
    chunk_index: int = Field(0, description="Position of this chunk in its parent document")
    page: int | None = Field(None, description="Page number (PDFs)")
    total_pages: int | None = Field(None, description="Total pages in the source document")

    # Language / Translation
    source_language: str | None = Field(
        None,
        description="BCP-47 language code of the source text (e.g. 'en', 'fr', 'sw')",
    )
    target_language: str | None = Field(
        None,
        description="BCP-47 language code of the translation target (e.g. 'en', 'fr', 'sw')",
    )
    language_pair: str | None = Field(
        None,
        description="Hyphenated source→target pair, e.g. 'fr-en'",
    )

    # Domain / Register
    domain: Literal[
        "general", "legal", "medical", "technical", "literary",
        "news", "conversational", "academic", "business"
    ] = Field("general", description="Subject domain for domain-adapted translation")
    text_register: Literal["formal", "informal", "neutral"] = Field(
        "neutral", alias="register", description="Linguistic register of the text"
    )

    # Ingestion provenance
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the chunk was stored",
    )
    doc_type: Literal["pdf", "txt", "docx", "csv", "url", "unknown"] = Field(
        "unknown", description="Original file type"
    )

    # Arbitrary extras (e.g. author, version, project_id)
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_chroma_dict(self) -> dict[str, Any]:
        """
        Flatten to a Chroma-compatible metadata dict (no nested objects).
        Chroma only supports str / int / float / bool values.
        """
        base = self.model_dump(exclude={"extra", "ingested_at"})
        base["ingested_at"] = self.ingested_at.isoformat()
        base.update({f"extra_{k}": str(v) for k, v in self.extra.items()})
        return {k: v for k, v in base.items() if v is not None}


# ── Self-Query Filter ─────────────────────────────────────────────────────────

class MetadataFilter(BaseModel):
    """
    Structured filter the LLM produces during self-querying retrieval.
    Fields mirror DocumentMetadata's filterable columns.
    """
    source_language: str | None = None
    target_language: str | None = None
    language_pair: str | None = None
    domain: str | None = None
    text_register: str | None = Field(None, alias="register")
    doc_type: str | None = None
    source: str | None = None   # filter by filename


# ── Inbound API ───────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Payload the client sends to /api/query."""
    query: str = Field(..., min_length=1, max_length=4096, description="User question")
    top_k: int = Field(5, ge=1, le=20, description="Number of retrieved chunks")
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    # Optional pre-built filter; if None the pipeline auto-generates one via self-querying
    metadata_filter: MetadataFilter | None = None
    # Translation-specific convenience fields
    source_language: str | None = Field(None, description="Source language override (BCP-47)")
    target_language: str | None = Field(None, description="Target language override (BCP-47)")
    preferred_provider: Literal["gemini", "anthropic", "auto"] = Field(
        "auto",
        description="Force a specific LLM provider, or let the pipeline decide",
    )


class IngestRequest(BaseModel):
    """Payload for ingesting a file path or URL."""
    source: str = Field(..., description="Absolute path or public URL")
    source_type: str = Field("pdf", description="pdf | txt | docx | url | csv")
    metadata: DocumentMetadata | dict[str, Any] = Field(default_factory=dict)


# ── Retrieved chunk ───────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    source: str
    page: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Convenience accessors derived from metadata
    @property
    def source_language(self) -> str | None:
        return self.metadata.get("source_language")

    @property
    def target_language(self) -> str | None:
        return self.metadata.get("target_language")

    @property
    def domain(self) -> str:
        return self.metadata.get("domain", "general")


# ── Outbound API ──────────────────────────────────────────────────────────────

class QueryResponse(BaseModel):
    """Full answer envelope returned to the client."""
    session_id: str
    query: str
    answer: str
    sources: list[RetrievedChunk]
    model: str
    provider: str = ""
    latency_ms: float
    # Language pair detected/used during this query
    detected_source_language: str | None = None
    detected_target_language: str | None = None
    # The self-generated metadata filter that was applied
    applied_filter: MetadataFilter | None = None
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class IngestResponse(BaseModel):
    status: str               # "ok" | "error"
    chunks_added: int
    source: str
    message: str = ""
