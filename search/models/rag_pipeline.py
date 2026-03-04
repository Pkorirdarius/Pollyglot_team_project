"""
search/models/rag_pipeline.py
──────────────────────────────
Translation-aware RAG pipeline with:
  • Dual-model routing  — Gemini primary, Anthropic fallback (or vice-versa)
  • Self-querying       — LLM generates a MetadataFilter before retrieval
  • Graceful fallback   — if primary fails, retries with the other provider
  • Token optimisation  — tight prompts, chunk truncation, low max_tokens

Fixes vs previous version:
  • Replaced deprecated google.generativeai with google.genai (new SDK)
  • Self-query uses a local regex classifier first — only calls LLM when needed,
    saving ~200 tokens per request
  • Context chunks truncated to 300 chars each (configurable) in the prompt
  • System prompts trimmed to essentials
  • Anthropic fallback model pinned to claude-haiku-4-5 (cheapest/fastest)
"""
from __future__ import annotations

import json
import re
import time
from uuid import uuid4

from loguru import logger

from config.settings import settings
from data_wrangling.vectorstore import similarity_search
from search.models.schemas import (
    MetadataFilter,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)

# ── Language map ──────────────────────────────────────────────────────────────

LANG_NAMES: dict[str, str] = {
    "en": "English", "fr": "French",  "es": "Spanish",  "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch",  "ru": "Russian",
    "zh": "Chinese", "ja": "Japanese",   "ko": "Korean", "ar": "Arabic",
    "sw": "Swahili", "hi": "Hindi",      "tr": "Turkish","pl": "Polish",
    "sv": "Swedish", "fi": "Finnish",    "da": "Danish", "no": "Norwegian",
}

# Regex patterns for fast local language detection (no LLM call needed)
_LANG_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r'\b(from|translate)\s+(\w+)\s+to\s+(\w+)', re.I), "from_to", ""),
    (re.compile(r'\bin\s+(french|spanish|german|italian|portuguese|arabic|chinese|japanese|korean|swahili|hindi|turkish|russian|dutch)\b', re.I), "in_lang", ""),
]

_LANG_NAME_TO_CODE = {v.lower(): k for k, v in LANG_NAMES.items()}
_DOMAIN_WORDS = {"legal", "medical", "technical", "literary", "news", "conversational", "academic", "business"}


def _lang_name(code: str | None) -> str:
    if not code:
        return "target language"
    return LANG_NAMES.get(code.lower(), code.upper())


# ── Fast local self-query (no LLM tokens spent) ───────────────────────────────

def _fast_extract_filter(query: str) -> MetadataFilter:
    q_lower = query.lower()
    mf = MetadataFilter()

    # IMPROVED: Handle "Translate X from French to English" AND "French to English translation"
    m = re.search(r'\b(?:from\s+)?(\w+)\s+to\s+(\w+)', q_lower)
    if m:
        src = _LANG_NAME_TO_CODE.get(m.group(1))
        tgt = _LANG_NAME_TO_CODE.get(m.group(2))
        if src: mf.source_language = src
        if tgt: mf.target_language = tgt

    # IMPROVED: Target language only (e.g., "How to say X in Spanish")
    if not mf.target_language:
        m = re.search(r'\bin\s+([a-zA-Z]+)(?:\s|$)', q_lower)
        if m:
            code = _LANG_NAME_TO_CODE.get(m.group(1))
            if code:
                mf.target_language = code

    # Logic for domain and register remains same...
    return mf

def _needs_llm_filter(query: str, fast_filter: MetadataFilter) -> bool:
    """
    Only call the LLM for self-querying if the fast filter found nothing
    AND the query looks complex enough to benefit from structured extraction.
    """
    has_something = any([
        fast_filter.source_language,
        fast_filter.target_language,
        fast_filter.domain,
    ])
    is_complex = len(query.split()) > 8
    return not has_something and is_complex


# ── Prompts (kept tight to minimise tokens) ───────────────────────────────────

# TOKEN OPTIMISATION: System prompt is the most expensive part (sent every call).
# Keep it to the essential rules only.
TRANSLATION_SYSTEM_PROMPT = (
    "You are Polyglot, an expert translation assistant. "
    "Rules: (1) Output the translation first. "
    "(2) Add a brief note only if there is genuine ambiguity or cultural nuance. "
    "(3) Use retrieved context for terminology consistency. "
    "(4) Never invent words — flag uncertainty with [?]. "
    "(5) Be concise."
)

# Self-query prompt — only used as LLM fallback when regex fails
SELF_QUERY_SYSTEM_PROMPT = (
    "Extract language/domain metadata from a translation query. "
    'Return ONLY a JSON object, e.g.: {"source_language":"fr","target_language":"en","domain":"legal"}. '
    "Use BCP-47 codes. Omit keys you cannot determine. No explanation."
)


# ── Prompt builder ────────────────────────────────────────────────────────────

# TOKEN OPTIMISATION: truncate each chunk to MAX_CHUNK_CHARS characters.
# A full 400-token chunk can be 1600+ chars; 300 chars ≈ 75 tokens.
MAX_CHUNK_CHARS = 300

def _build_prompt(
    query: str,
    chunks: list[RetrievedChunk],
    source_lang: str | None,
    target_lang: str | None,
) -> str:
    parts: list[str] = []

    if chunks:
        ctx_lines = []
        for c in chunks:
            # Truncate chunk text to save tokens while keeping the gist
            text = c.text[:MAX_CHUNK_CHARS] + ("…" if len(c.text) > MAX_CHUNK_CHARS else "")
            ctx_lines.append(f"[{c.source}|{c.domain}] {text}")
        parts.append("Context:\n" + "\n---\n".join(ctx_lines))

    if source_lang or target_lang:
        parts.append(f"Direction: {_lang_name(source_lang)} → {_lang_name(target_lang)}")

    parts.append(f"Request: {query}")
    return "\n\n".join(parts)


# ── LLM callers ───────────────────────────────────────────────────────────────

def _call_gemini(system: str, user: str) -> str:
    """Use the new google-genai SDK (google.generativeai is deprecated)."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)
    response = client.models.generate_content(
        model=settings.llm_model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=settings.max_tokens,
            temperature=settings.temperature,
        ),
    )
    return response.text


def _call_anthropic(system: str, user: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    # TOKEN OPTIMISATION: use Haiku as the Anthropic fallback — same quality
    # for translation tasks at ~10x lower cost than Sonnet.
    fallback_model = (
        settings.llm_model
        if settings.llm_provider == "anthropic"
        else "claude-haiku-4-5-20251001"
    )
    response = client.messages.create(
        model=fallback_model,
        max_tokens=settings.max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


def _generate_with_fallback(
    system: str,
    user: str,
    preferred: str = "auto",
) -> tuple[str, str]:
    """
    Call the preferred provider; fall back to the other on failure.
    Returns (answer_text, provider_name).
    """
    primary   = preferred if preferred != "auto" else settings.llm_provider
    secondary = "anthropic" if primary == "gemini" else "gemini"

    callers = {
        "gemini":    (_call_gemini,    bool(settings.gemini_api_key)),
        "anthropic": (_call_anthropic, bool(settings.anthropic_api_key)),
    }

    for provider in [primary, secondary]:
        caller, available = callers.get(provider, (None, False))
        if not available or caller is None:
            logger.warning(f"Provider '{provider}' skipped — no API key")
            continue
        try:
            logger.info(f"Calling provider: {provider}")
            return caller(system, user), provider
        except Exception as exc:
            logger.warning(f"Provider '{provider}' failed: {exc}. Trying fallback…")

    raise RuntimeError(
        "All LLM providers failed. "
        "Check GEMINI_API_KEY and ANTHROPIC_API_KEY in your .env file."
    )


# ── Self-querying (LLM fallback only) ────────────────────────────────────────

def _extract_metadata_filter(query: str) -> MetadataFilter:
    """
    TOKEN OPTIMISATION:
      1. Try fast regex extraction first (0 tokens).
      2. Only call the LLM if regex found nothing AND query is complex.
    """
    fast = _fast_extract_filter(query)
    if not _needs_llm_filter(query, fast):
        logger.debug("Filter extracted via regex (0 tokens spent)")
        return fast

    # LLM fallback — costs ~100-150 tokens
    logger.debug("Regex insufficient — using LLM for filter extraction")
    try:
        raw, _ = _generate_with_fallback(
            system=SELF_QUERY_SYSTEM_PROMPT,
            user=query,
            preferred="auto",
        )
        m = re.search(r"\{.*?\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            # Merge LLM result with regex result (regex wins on conflicts)
            merged = {k: v for k, v in data.items() if v}
            if fast.source_language: merged["source_language"] = fast.source_language
            if fast.target_language: merged["target_language"] = fast.target_language
            if fast.domain:          merged["domain"]          = fast.domain
            return MetadataFilter(**merged)
    except Exception as exc:
        logger.warning(f"LLM filter extraction failed: {exc}")

    return fast  # fall back to whatever regex found


# ── Chroma where-clause builder ───────────────────────────────────────────────

def _filter_to_chroma_where(f: MetadataFilter) -> dict | None:
    # Build a simple dict. Chroma handles multiple keys as an implicit AND.
    where = {}
    
    # Map pydantic model fields to metadata keys
    # Ensure keys match exactly what was stored during ingest_documents
    if f.source_language:
        where["source_language"] = f.source_language
    if f.target_language:
        where["target_language"] = f.target_language
    if f.domain:
        where["domain"] = f.domain
    if f.text_register:
        where["text_register"] = f.text_register

    return where if where else None

# ── Public entry point ────────────────────────────────────────────────────────

def run_rag_query(request: QueryRequest) -> QueryResponse:
    """
    Full RAG pipeline:
      1. Self-query  — extract MetadataFilter (regex first, LLM only if needed)
      2. Retrieve    — similarity search with optional metadata filter
      3. Generate    — dual-model with fallback
      4. Return      — structured QueryResponse
    """
    t0         = time.perf_counter()
    session_id = request.session_id or str(uuid4())
    logger.info(f"[{session_id}] Query: {request.query!r}")

    # 1. Metadata filter
    if request.metadata_filter:
        mf = request.metadata_filter
    else:
        mf = _extract_metadata_filter(request.query)

    # Apply explicit overrides from the request
    # Only override if the request explicitly provides a non-empty string
    if request.source_language and request.source_language.strip():
        mf.source_language = request.source_language
    if request.target_language and request.target_language.strip():
        mf.target_language = request.target_language
    if mf.source_language and mf.target_language:
        mf.language_pair = f"{mf.source_language}-{mf.target_language}"

    logger.debug(f"[{session_id}] Filter: {mf.model_dump(exclude_none=True)}")

    # 2. Retrieval
    where = _filter_to_chroma_where(mf)
    raw   = similarity_search(query=request.query, top_k=request.top_k, where=where)

    # Retry without filter if nothing came back
    if not raw and where:
        logger.info(f"[{session_id}] No filtered results — retrying without filter")
        raw = similarity_search(query=request.query, top_k=request.top_k)

    sources: list[RetrievedChunk] = [
        RetrievedChunk(
            chunk_id=str(uuid4()),
            text=doc.page_content,
            score=score,
            source=doc.metadata.get("source", "unknown"),
            page=doc.metadata.get("page"),
            metadata=doc.metadata,
        )
        for doc, score in raw
    ]
    logger.info(f"[{session_id}] {len(sources)} chunks retrieved")

    # 3. Generation
    prompt = _build_prompt(
        query=request.query,
        chunks=sources,
        source_lang=mf.source_language,
        target_lang=mf.target_language,
    )
    answer, provider = _generate_with_fallback(
        system=TRANSLATION_SYSTEM_PROMPT,
        user=prompt,
        preferred=request.preferred_provider,
    )

    ms = (time.perf_counter() - t0) * 1000
    logger.success(f"[{session_id}] Done in {ms:.0f} ms via {provider}")

    return QueryResponse(
        session_id=session_id,
        query=request.query,
        answer=answer,
        sources=sources,
        model=settings.llm_model,
        provider=provider,
        latency_ms=ms,
        detected_source_language=mf.source_language,
        detected_target_language=mf.target_language,
        applied_filter=mf,
    )
