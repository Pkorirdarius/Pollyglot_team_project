"""
models/rag_pipeline.py
──────────────────────
Core RAG pipeline: retrieve relevant chunks then generate an answer.

Works with Anthropic Claude or OpenAI GPT — toggled via LLM_PROVIDER env var.
"""

from __future__ import annotations

import time
from uuid import uuid4

from loguru import logger

from config.settings import settings
from data_wrangling.vectorstore import similarity_search
from models.schemas import QueryRequest, QueryResponse, RetrievedChunk

# ── Prompt template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions
strictly based on the provided context. If the context does not contain enough
information to answer the question, say so honestly instead of making things up.

Rules:
- Ground every claim in the retrieved context.
- Cite the source file/page when relevant.
- Be concise but complete.
- Use markdown for structure when helpful.
"""

def _build_user_prompt(query: str, context_chunks: list[RetrievedChunk]) -> str:
    context_text = "\n\n---\n\n".join(
        f"[Source: {c.source}, score={c.score:.2f}]\n{c.text}"
        for c in context_chunks
    )
    return (
        f"## Retrieved Context\n\n{context_text}\n\n"
        f"## Question\n\n{query}\n\n"
        f"## Answer"
    )


# ── LLM callers ───────────────────────────────────────────────────────────────

def _call_anthropic(system: str, user: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=settings.max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


def _call_openai(system: str, user: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.llm_model,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content


def _generate(system: str, user: str) -> str:
    if settings.llm_provider == "anthropic":
        return _call_anthropic(system, user)
    elif settings.llm_provider == "openai":
        return _call_openai(system, user)
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


# ── Public pipeline entry point ───────────────────────────────────────────────

def run_rag_query(request: QueryRequest) -> QueryResponse:
    """
    Full RAG pipeline:
      1. Embed the query and retrieve relevant chunks from the vector store.
      2. Build a context-aware prompt.
      3. Call the LLM and return a structured QueryResponse.
    """
    t0 = time.perf_counter()
    session_id = request.session_id or str(uuid4())

    logger.info(f"[{session_id}] RAG query: {request.query!r}")

    # 1. Retrieve
    raw_results = similarity_search(
        query=request.query,
        top_k=request.top_k,
    )

    sources: list[RetrievedChunk] = [
        RetrievedChunk(
            chunk_id=str(uuid4()),
            text=doc.page_content,
            score=score,
            source=doc.metadata.get("source", "unknown"),
            page=doc.metadata.get("page"),
            metadata=doc.metadata,
        )
        for doc, score in raw_results
    ]

    logger.info(f"[{session_id}] Retrieved {len(sources)} chunks")

    # 2. Generate
    user_prompt = _build_user_prompt(request.query, sources)
    answer = _generate(SYSTEM_PROMPT, user_prompt)

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.success(f"[{session_id}] Answer generated in {latency_ms:.0f} ms")

    return QueryResponse(
        session_id=session_id,
        query=request.query,
        answer=answer,
        sources=sources,
        model=settings.llm_model,
        latency_ms=latency_ms,
    )
