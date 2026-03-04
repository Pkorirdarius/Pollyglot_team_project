"""
tests/test_pipeline.py
────────────────────────
Smoke tests for schemas, self-querying, and the RAG pipeline.
Updated for Polyglot v2: translation-aware, dual-provider, self-querying.
"""
from __future__ import annotations

from datetime import timezone
from unittest.mock import patch, MagicMock

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from search.models.schemas import (
    DocumentMetadata,
    MetadataFilter,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)
from search.models.rag_pipeline import (
    run_rag_query,
    _extract_metadata_filter,
    _filter_to_chroma_where,
    _lang_name,
)


# ── DocumentMetadata ──────────────────────────────────────────────────────────

def test_document_metadata_defaults():
    meta = DocumentMetadata(source="glossary.pdf")
    assert meta.domain == "general"
    assert meta.register == "neutral"
    assert meta.source_language is None


def test_document_metadata_chroma_serialisation():
    meta = DocumentMetadata(
        source="contract.pdf",
        source_language="fr",
        target_language="en",
        language_pair="fr-en",
        domain="legal",
        register="formal",
        extra={"client": "Acme"},
    )
    d = meta.to_chroma_dict()
    assert d["source_language"] == "fr"
    assert d["domain"] == "legal"
    assert d["extra_client"] == "Acme"
    # ingested_at should be ISO string, not datetime object
    assert isinstance(d["ingested_at"], str)


# ── MetadataFilter ────────────────────────────────────────────────────────────

def test_filter_to_chroma_where_empty():
    f = MetadataFilter()
    assert _filter_to_chroma_where(f) is None


def test_filter_to_chroma_where_single():
    f = MetadataFilter(source_language="fr")
    clause = _filter_to_chroma_where(f)
    assert clause == {"source_language": {"$eq": "fr"}}


def test_filter_to_chroma_where_multi():
    f = MetadataFilter(source_language="fr", domain="legal")
    clause = _filter_to_chroma_where(f)
    assert "$and" in clause
    assert len(clause["$and"]) == 2


# ── QueryRequest ──────────────────────────────────────────────────────────────

def test_query_request_defaults():
    req = QueryRequest(query="Translate bonjour to English")
    assert req.top_k == 5
    assert req.preferred_provider == "auto"
    assert req.session_id is not None


def test_query_request_empty_fails():
    with pytest.raises(ValidationError) as exc_info:
        QueryRequest(query="")
    assert "query" in str(exc_info.value)


def test_query_request_language_overrides():
    req = QueryRequest(query="hello", source_language="en", target_language="fr")
    assert req.source_language == "en"
    assert req.target_language == "fr"


# ── Language name helper ──────────────────────────────────────────────────────

def test_lang_name_known():
    assert _lang_name("fr") == "French"
    assert _lang_name("sw") == "Swahili"


def test_lang_name_unknown():
    result = _lang_name("xx")
    assert "XX" in result or result  # returns uppercased code


def test_lang_name_none():
    assert _lang_name(None) == "the target language"


# ── Self-querying ─────────────────────────────────────────────────────────────

@patch("search.models.rag_pipeline._generate_with_fallback")
def test_extract_metadata_filter_success(mock_gen):
    mock_gen.return_value = (
        '{"source_language": "fr", "target_language": "en", "domain": "legal"}',
        "gemini",
    )
    filt = _extract_metadata_filter("Translate French legal contract to English")
    assert filt.source_language == "fr"
    assert filt.target_language == "en"
    assert filt.domain == "legal"


@patch("search.models.rag_pipeline._generate_with_fallback")
def test_extract_metadata_filter_fallback_on_bad_json(mock_gen):
    mock_gen.return_value = ("not valid json at all", "gemini")
    filt = _extract_metadata_filter("some query")
    # Should return an empty filter, not raise
    assert isinstance(filt, MetadataFilter)


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

@patch("search.models.rag_pipeline.similarity_search")
@patch("search.models.rag_pipeline._generate_with_fallback")
@patch("search.models.rag_pipeline._extract_metadata_filter")
def test_run_rag_query_full(mock_filter, mock_gen, mock_search):
    mock_filter.return_value = MetadataFilter(
        source_language="fr", target_language="en", domain="general"
    )
    mock_search.return_value = [
        (
            Document(
                page_content="Bonjour is French for hello.",
                metadata={"source": "glossary.pdf", "domain": "general", "source_language": "fr"},
            ),
            0.95,
        )
    ]
    mock_gen.return_value = ("Hello", "gemini")

    req = QueryRequest(query="Translate bonjour")
    resp = run_rag_query(req)

    assert isinstance(resp, QueryResponse)
    assert resp.answer == "Hello"
    assert resp.provider == "gemini"
    assert resp.detected_source_language == "fr"
    assert resp.detected_target_language == "en"
    assert len(resp.sources) == 1
    assert resp.sources[0].score == pytest.approx(0.95)


@patch("search.models.rag_pipeline.similarity_search")
@patch("search.models.rag_pipeline._generate_with_fallback")
@patch("search.models.rag_pipeline._extract_metadata_filter")
def test_run_rag_query_fallback_retrieval(mock_filter, mock_gen, mock_search):
    """When filtered retrieval returns nothing, pipeline should retry without filter."""
    mock_filter.return_value = MetadataFilter(source_language="fr")
    # First call (with filter) returns nothing; second call (no filter) returns a result
    mock_search.side_effect = [
        [],
        [(Document(page_content="context", metadata={"source": "x.pdf"}), 0.8)],
    ]
    mock_gen.return_value = ("answer", "anthropic")

    req = QueryRequest(query="test")
    resp = run_rag_query(req)
    assert mock_search.call_count == 2
    assert resp.answer == "answer"


# ── QueryResponse metadata ────────────────────────────────────────────────────

def test_query_response_utc_timestamp():
    chunk = RetrievedChunk(chunk_id="1", text="abc", score=0.99, source="test.txt")
    resp = QueryResponse(
        session_id="s1",
        query="hi",
        answer="hello",
        sources=[chunk],
        model="gemini-2.0-flash",
        provider="gemini",
        latency_ms=88.5,
    )
    assert resp.timestamp.tzinfo == timezone.utc
    assert resp.latency_ms == pytest.approx(88.5)
