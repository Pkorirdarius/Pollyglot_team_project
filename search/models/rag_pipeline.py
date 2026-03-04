"""
search/models/rag_pipeline.py
──────────────────────────────
Translation-aware RAG pipeline with:
  • Dual-model routing  — Gemini (primary) + Anthropic Claude (secondary / fallback)
  • Self-querying       — LLM generates a MetadataFilter before retrieval
  • Language detection  — Auto-detects source language when not supplied
  • Graceful fallback   — If primary provider fails, retries with the other
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

# ── Language utilities ────────────────────────────────────────────────────────

# Common BCP-47 codes → human-readable names (expand as needed)
LANG_NAMES: dict[str, str] = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
    "sw": "Swahili", "hi": "Hindi", "tr": "Turkish", "pl": "Polish",
    "sv": "Swedish", "fi": "Finnish", "da": "Danish", "no": "Norwegian",
}


def _lang_name(code: str | None) -> str:
    if not code:
        return "the target language"
    return LANG_NAMES.get(code.lower(), code.upper())


# ── System prompts ────────────────────────────────────────────────────────────

TRANSLATION_SYSTEM_PROMPT = """You are Polyglot, an expert multilingual translation assistant.

Your capabilities:
- Translate text accurately between any language pair
- Adapt register (formal / informal / neutral) as requested
- Respect domain-specific terminology (legal, medical, technical, etc.)
- Provide concise explanations of grammar, idiom, or cultural nuance when helpful
- Use the retrieved context (glossaries, parallel texts, style guides) to ensure consistency

Rules:
- Always produce the translation first, then optional notes
- Ground terminology choices in the retrieved context when available
- Never fabricate words or invent proper nouns
- If unsure about a term, flag it with [uncertain: …]
- Format responses in Markdown when structure helps clarity
"""

SELF_QUERY_SYSTEM_PROMPT = """You extract structured metadata filters from a user's translation query.

Return ONLY a valid JSON object with these optional keys (omit keys you cannot determine):
{
  "source_language": "<BCP-47 code or null>",
  "target_language": "<BCP-47 code or null>",
  "language_pair": "<src-tgt or null>",
  "domain": "<general|legal|medical|technical|literary|news|conversational|academic|business or null>",
  "register": "<formal|informal|neutral or null>",
  "doc_type": "<pdf|txt|docx|csv|url|unknown or null>",
  "source": "<filename if specifically mentioned or null>"
}

Examples:
User: "Translate this legal contract from French to English formally"
→ {"source_language":"fr","target_language":"en","language_pair":"fr-en","domain":"legal","register":"formal"}

User: "How do I say hello in Spanish?"
→ {"source_language":"en","target_language":"es","language_pair":"en-es","domain":"conversational","register":"informal"}
"""


def _build_translation_prompt(
    query: str,
    chunks: list[RetrievedChunk],
    source_lang: str | None,
    target_lang: str | None,
) -> str:
    src = _lang_name(source_lang)
    tgt = _lang_name(target_lang)

    context_block = ""
    if chunks:
        parts = [
            f"[Source: {c.source} | score={c.score:.2f} | domain={c.domain}]\n{c.text}"
            for c in chunks
        ]
        context_block = "## Reference Context\n\n" + "\n\n---\n\n".join(parts) + "\n\n"

    lang_hint = ""
    if source_lang or target_lang:
        lang_hint = f"**Translation direction:** {src} → {tgt}\n\n"

    return (
        f"{context_block}"
        f"{lang_hint}"
        f"## Request\n\n{query}\n\n"
        f"## Response"
    )


# ── LLM callers ───────────────────────────────────────────────────────────────

def _call_gemini(system: str, user: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(
        model_name=settings.llm_model,
        system_instruction=system,
    )
    response = model.generate_content(
        user,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=settings.max_tokens,
            temperature=settings.temperature,
        ),
    )
    return response.text


def _call_anthropic(system: str, user: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.llm_model
        if settings.llm_provider == "anthropic"
        else "claude-sonnet-4-20250514",  # fallback model when Gemini is primary
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
    Returns (answer_text, provider_used).
    Routing logic:
      preferred="gemini"    → try Gemini only
      preferred="anthropic" → try Anthropic only
      preferred="auto"      → use settings.llm_provider as primary, other as fallback
    """
    primary = preferred if preferred != "auto" else settings.llm_provider
    secondary = "anthropic" if primary == "gemini" else "gemini"

    callers = {
        "gemini": (_call_gemini, bool(settings.gemini_api_key)),
        "anthropic": (_call_anthropic, bool(settings.anthropic_api_key)),
    }

    for provider in [primary, secondary]:
        caller, available = callers.get(provider, (None, False))
        if not available or caller is None:
            logger.warning(f"Provider {provider!r} skipped — API key not configured")
            continue
        try:
            logger.info(f"Calling provider: {provider}")
            answer = caller(system, user)
            return answer, provider
        except Exception as exc:
            logger.warning(f"Provider {provider!r} failed: {exc}. Trying fallback…")

    raise RuntimeError(
        "All configured LLM providers failed. "
        "Check your API keys in .env (GEMINI_API_KEY / ANTHROPIC_API_KEY)."
    )


# ── Self-querying ─────────────────────────────────────────────────────────────

def _extract_metadata_filter(query: str) -> MetadataFilter:
    """
    Ask the LLM to parse the user query into a MetadataFilter.
    Falls back to an empty filter if parsing fails.
    """
    try:
        raw, _ = _generate_with_fallback(
            system=SELF_QUERY_SYSTEM_PROMPT,
            user=query,
            preferred="auto",
        )
        # Extract JSON even if surrounded by markdown fences
        json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return MetadataFilter(**{k: v for k, v in data.items() if v is not None})
    except Exception as exc:
        logger.warning(f"Self-querying filter extraction failed: {exc}")
    return MetadataFilter()


def _filter_to_chroma_where(f: MetadataFilter) -> dict | None:
    """Convert MetadataFilter to a Chroma $where clause."""
    clauses: list[dict] = []
    for field, value in f.model_dump().items():
        if value is not None:
            clauses.append({field: {"$eq": value}})
    if not clauses:
        return None
    return {"$and": clauses} if len(clauses) > 1 else clauses[0]


# ── Public pipeline entry point ───────────────────────────────────────────────

def run_rag_query(request: QueryRequest) -> QueryResponse:
    """
    Full translation-aware RAG pipeline:
      1. Self-querying  — extract MetadataFilter from the raw query
      2. Retrieval      — similarity search with optional metadata filter
      3. Generation     — dual-model routing with automatic fallback
      4. Response       — structured QueryResponse with provenance
    """
    t0 = time.perf_counter()
    session_id = request.session_id or str(uuid4())
    logger.info(f"[{session_id}] Query: {request.query!r}")

    # ── 1. Self-querying: derive metadata filter ──────────────────────────────
    if request.metadata_filter:
        mf = request.metadata_filter
        logger.debug(f"[{session_id}] Using client-supplied metadata filter")
    else:
        mf = _extract_metadata_filter(request.query)
        logger.debug(f"[{session_id}] Self-queried filter: {mf.model_dump(exclude_none=True)}")

    # Honour explicit language overrides from the request
    if request.source_language:
        mf.source_language = request.source_language
    if request.target_language:
        mf.target_language = request.target_language
    if mf.source_language and mf.target_language and not mf.language_pair:
        mf.language_pair = f"{mf.source_language}-{mf.target_language}"

    # ── 2. Retrieval ──────────────────────────────────────────────────────────
    where_clause = _filter_to_chroma_where(mf)
    raw_results = similarity_search(
        query=request.query,
        top_k=request.top_k,
        where=where_clause,          # pass filter to vectorstore helper
    )

    # If filtered retrieval returns nothing, retry without filter
    if not raw_results and where_clause:
        logger.info(f"[{session_id}] No results with filter — retrying without filter")
        raw_results = similarity_search(query=request.query, top_k=request.top_k)

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

    # ── 3. Generation ─────────────────────────────────────────────────────────
    user_prompt = _build_translation_prompt(
        query=request.query,
        chunks=sources,
        source_lang=mf.source_language,
        target_lang=mf.target_language,
    )
    answer, provider_used = _generate_with_fallback(
        system=TRANSLATION_SYSTEM_PROMPT,
        user=user_prompt,
        preferred=request.preferred_provider,
    )

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.success(
        f"[{session_id}] Answer in {latency_ms:.0f} ms via {provider_used}"
    )

    return QueryResponse(
        session_id=session_id,
        query=request.query,
        answer=answer,
        sources=sources,
        model=settings.llm_model,
        provider=provider_used,
        latency_ms=latency_ms,
        detected_source_language=mf.source_language,
        detected_target_language=mf.target_language,
        applied_filter=mf,
    )
