"""
tests/test_pipeline.py
Updated for 2026: Pydantic V2 validation and asynchronous pipeline testing.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from pydantic import ValidationError
from datetime import datetime, timezone
from search.models.schemas import QueryRequest, QueryResponse ,RetrievedChunk
from search.models.rag_pipeline import run_rag_query
from langchain_core.documents import Document


# ── Model Validation Tests ───────────────────────────────────────────────────

def test_query_request_validation():
    """Ensure QueryRequest enforces constraints (min_length, default top_k)."""
    req = QueryRequest(query="What is RAG?", top_k=3)
    assert req.query == "What is RAG?"
    assert req.top_k == 3
    assert req.session_id is not None


def test_query_request_empty_fails():
    """Pydantic V2 should raise ValidationError if query is empty or too short."""
    with pytest.raises(ValidationError) as exc_info:
        QueryRequest(query="", top_k=3)
    
    # Verify the error is specifically about the 'query' field
    assert "query" in str(exc_info.value)


# ── Pipeline Logic Tests ─────────────────────────────────────────────────────

@patch("models.rag_pipeline.similarity_search")
@patch("models.rag_pipeline._generate")
def test_run_rag_query(mock_gen, mock_search):
    """Smoke test for the core RAG logic using mocked components."""

    # Mock the Vector Store retrieval
    mock_search.return_value = [
        (Document(page_content="Some context.", metadata={"source": "doc.pdf"}), 0.91)
    ]
    
    # Mock the LLM generation
    mock_gen.return_value = "This is the generated answer."

    req = QueryRequest(query="What does the document say?")
    resp = run_rag_query(req)

    # Assertions
    assert isinstance(resp, QueryResponse)
    assert resp.answer == "This is the generated answer."
    assert len(resp.sources) == 1
    assert resp.sources[0].score == pytest.approx(0.91)
    assert resp.sources[0].source == "doc.pdf"


# ── Performance & Metadata Tests ─────────────────────────────────────────────

def test_query_response_latency_tracking():
    """Ensure the response includes valid metadata like latency and timestamp."""
    

    # Manually creating a response to test schema defaults
    chunk = RetrievedChunk(chunk_id="1", text="abc", score=0.99, source="test.txt")
    resp = QueryResponse(
        session_id="test-session",
        query="hi",
        answer="hello",
        sources=[chunk],
        model="gemini-2.0-flash",
        latency_ms=150.5
    )
    
    assert resp.latency_ms == 150.5
    # Ensure our 2026 UTC fix is working
    assert resp.timestamp.tzinfo == timezone.utc