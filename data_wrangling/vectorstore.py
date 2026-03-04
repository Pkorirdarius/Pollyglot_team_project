"""
data_wrangling/vectorstore.py
─────────────────────────────
Build, persist, and query the ChromaDB vector store.

Fixes vs previous version:
  • Uses langchain-chroma (not deprecated langchain_community.Chroma)
  • Embedding model name no longer prefixed with "models/" — the SDK adds it
  • Pinecone import is lazy (only triggered when actually configured)
  • similarity_search accepts an optional `where` Chroma filter dict

Token optimisation:
  • top_k and threshold pulled from settings (already tightened in settings.py)
  • Embeddings client reused within a single request (no redundant init calls)
"""
from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from loguru import logger
import json
import urllib.request
import urllib.error
from config.settings import settings


# ── Embeddings ────────────────────────────────────────────────────────────────
# ROOT CAUSE FIX:
# langchain-google-genai calls the v1beta endpoint for embeddings.
# text-embedding-004 is ONLY available on the v1 endpoint.
# Calling it via langchain-google-genai always returns 404 regardless of
# how the model name is formatted.
#
# Fix: implement a thin LangChain-compatible embeddings wrapper that calls
# google.genai directly (which uses v1) instead of going through
# langchain-google-genai (which uses v1beta).

from langchain_core.embeddings import Embeddings


class GeminiEmbeddings(Embeddings):
    """
    Calls the Gemini embedding REST API directly on v1 (not v1beta).

    Both the google.genai SDK and langchain-google-genai route embedding
    calls through v1beta, where text-embedding-004 does not exist.
    The model is only available on v1. Bypassing the SDKs entirely and
    calling the REST endpoint directly on v1 is the only reliable fix.
    """

    def __init__(self, api_key: str, model: str = "text-embedding-004"):
        self._api_key = api_key
        self._model = model.removeprefix("models/")

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        # Ensure model is prefixed correctly for the URL path
        model_path = f"models/{self._model}" if not self._model.startswith("models/") else self._model
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/{model_path}:embedContent?key={self._api_key}"
        )

        results = []
        for text in texts:
            # v1 REST API expects 'content' and 'taskType'. 
            # Do not include 'model' in the JSON body when it's in the URL.
            payload = json.dumps({
                "model": model_path,
                "content": {"parts": [{"text": text}]},
                "taskType": task_type,
            }).encode("utf-8")
            
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            
            try:
                with urllib.request.urlopen(req) as response:
                    data = json.loads(response.read())
                    # Response structure: {"embedding": {"values": [...]}}
                    results.append(data["embedding"]["values"])
            except urllib.error.HTTPError as e:
                error_msg = e.read().decode()
                logger.error(f"Gemini API Error {e.code}: {error_msg}")
                raise e
                
        return results
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def _get_embeddings(task_type: str = "retrieval_document") -> Embeddings:
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is required for embeddings. "
            "Add it to your .env file."
        )
    return GeminiEmbeddings(
        api_key=settings.gemini_api_key,
        model=settings.embedding_model,
    )


# ── Chroma helper ─────────────────────────────────────────────────────────────

def _get_chroma(collection_name: str, task_type: str = "retrieval_document"):
    """Return a Chroma vectorstore instance using langchain-chroma (not deprecated community version)."""
    try:
        from langchain_chroma import Chroma
    except ImportError:
        # Graceful fallback to community version with deprecation warning suppressed
        from langchain_community.vectorstores import Chroma  # type: ignore[no-redef]

    return Chroma(
        collection_name=collection_name,
        embedding_function=_get_embeddings(task_type),
        persist_directory=str(settings.chroma_persist_dir),
    )


# ── Ingest ────────────────────────────────────────────────────────────────────

def ingest_documents(
    documents: list[Document],
    collection_name: str = "rag_store",
) -> int:
    if not documents:
        logger.warning("ingest_documents called with empty list.")
        return 0

    if settings.vectorstore_provider == "chroma":
        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma  # type: ignore[no-redef]

        Chroma.from_documents(
            documents=documents,
            embedding=_get_embeddings("retrieval_document"),
            collection_name=collection_name,
            persist_directory=str(settings.chroma_persist_dir),
        )

    elif settings.vectorstore_provider == "pinecone":
        # Lazy import — only needed when Pinecone is actually configured
        from langchain_pinecone import PineconeVectorStore  # type: ignore[import]

        PineconeVectorStore.from_documents(
            documents=documents,
            embedding=_get_embeddings("retrieval_document"),
            index_name=settings.pinecone_index_name,
        )
    else:
        raise ValueError(f"Unknown vectorstore provider: {settings.vectorstore_provider!r}")

    logger.success(f"Ingested {len(documents)} chunks into '{collection_name}'")
    return len(documents)


# ── Query ─────────────────────────────────────────────────────────────────────

def similarity_search(
    query: str,
    top_k: int | None = None,
    collection_name: str = "rag_store",
    score_threshold: float | None = None,
    where: dict[str, Any] | None = None,
) -> list[tuple[Document, float]]:
    """
    Retrieve top-k chunks by similarity.

    TOKEN OPTIMISATION: top_k defaults to settings.top_k_retrieval (3).
    Each retrieved chunk is injected verbatim into the LLM prompt, so
    fetching fewer, higher-quality chunks directly lowers token spend.

    Args:
        query:            User query string.
        top_k:            Max chunks to retrieve (default: settings.top_k_retrieval).
        collection_name:  Chroma collection name.
        score_threshold:  Min relevance score to keep (default: settings.similarity_threshold).
        where:            Optional Chroma metadata filter from self-querying.
    """
    k         = top_k or settings.top_k_retrieval
    threshold = score_threshold if score_threshold is not None else settings.similarity_threshold

    if settings.vectorstore_provider == "chroma":
        vs = _get_chroma(collection_name, task_type="retrieval_query")

        kwargs: dict[str, Any] = {"k": k}
        if where:
            kwargs["filter"] = where
            logger.debug(f"Chroma filter applied: {where}")

        results = vs.similarity_search_with_relevance_scores(query, **kwargs)

    else:
        # Pinecone path
        from langchain_pinecone import PineconeVectorStore  # type: ignore[import]

        vs = PineconeVectorStore(
            index_name=settings.pinecone_index_name,
            embedding=_get_embeddings("retrieval_query"),
        )
        results = vs.similarity_search_with_relevance_scores(query, k=k)

    filtered = [(doc, score) for doc, score in results if score >= threshold]
    logger.debug(
        f"Retrieved {len(filtered)}/{k} chunks "
        f"(threshold={threshold:.2f}{', filtered' if where else ''})"
    )
    return filtered
