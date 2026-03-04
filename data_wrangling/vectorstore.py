"""
data_wrangling/vectorstore.py
─────────────────────────────
Build, persist, and query a vector store.
Supports: ChromaDB (local) and Pinecone (cloud).

2026 updates:
  • similarity_search accepts an optional `where` Chroma filter dict
  • Embeddings helper selects provider from settings (Gemini / OpenAI)
  • Pinecone path updated to langchain-pinecone ≥ 0.2
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from loguru import logger
# OpenAI fallback
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from config.settings import settings

if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStore


# ── Embeddings ────────────────────────────────────────────────────────────────

def _get_embeddings(task_type: str = "retrieval_document"):
    """Return an embeddings instance configured for the current provider."""
    if settings.llm_provider in ("gemini", "anthropic"):
        # Gemini embeddings work for both provider modes
        # (Anthropic has no native embedding model, so we reuse Gemini)
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is required for embeddings even when "
                "LLM_PROVIDER=anthropic. Add it to your .env file."
            )
        return GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.gemini_api_key,
            task_type=task_type,
        )

    
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )


# ── Vector store factory ──────────────────────────────────────────────────────

def get_vectorstore(
    collection_name: str = "rag_store",
    task_type: str = "retrieval_document",
) -> "VectorStore":
    embeddings = _get_embeddings(task_type)

    if settings.vectorstore_provider == "chroma":
        from langchain_community.vectorstores import Chroma

        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(settings.chroma_persist_dir),
        )

    if settings.vectorstore_provider == "pinecone":
        
        return PineconeVectorStore(
            index_name=settings.pinecone_index_name,
            embedding=embeddings,
        )

    raise ValueError(f"Unknown vectorstore provider: {settings.vectorstore_provider!r}")


# ── Ingest ────────────────────────────────────────────────────────────────────

def ingest_documents(
    documents: list[Document],
    collection_name: str = "rag_store",
) -> int:
    if not documents:
        logger.warning("ingest_documents called with empty list.")
        return 0

    embeddings = _get_embeddings("retrieval_document")

    if settings.vectorstore_provider == "chroma":
        from langchain_community.vectorstores import Chroma

        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=str(settings.chroma_persist_dir),
        )

    elif settings.vectorstore_provider == "pinecone":
        
        PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=settings.pinecone_index_name,
        )
    else:
        raise ValueError(f"Unknown vectorstore provider: {settings.vectorstore_provider!r}")

    logger.success(f"Ingested {len(documents)} chunks into {collection_name!r}")
    return len(documents)


# ── Query ─────────────────────────────────────────────────────────────────────

def similarity_search(
    query: str,
    top_k: int | None = None,
    collection_name: str = "rag_store",
    score_threshold: float | None = None,
    where: dict[str, Any] | None = None,        # ← self-querying filter
) -> list[tuple[Document, float]]:
    """
    Retrieve top-k chunks ranked by similarity.

    Args:
        query:            The user query string.
        top_k:            Number of results to retrieve (default: settings.top_k_retrieval).
        collection_name:  Chroma collection to query.
        score_threshold:  Minimum relevance score to keep (default: settings.similarity_threshold).
        where:            Optional Chroma $where filter dict produced by self-querying.

    Returns:
        List of (Document, score) tuples above the threshold.
    """
    k = top_k or settings.top_k_retrieval
    threshold = score_threshold or settings.similarity_threshold

    embeddings = _get_embeddings("retrieval_query")

    if settings.vectorstore_provider == "chroma":
        from langchain_community.vectorstores import Chroma

        vs = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(settings.chroma_persist_dir),
        )

        kwargs: dict[str, Any] = {"k": k}
        if where:
            kwargs["filter"] = where
            logger.debug(f"Applying Chroma filter: {where}")

        results = vs.similarity_search_with_relevance_scores(query, **kwargs)

    else:
        # Pinecone / generic path — metadata filtering passed via search_kwargs
        vs = get_vectorstore(collection_name, task_type="retrieval_query")
        results = vs.similarity_search_with_relevance_scores(query, k=k)

    filtered = [(doc, score) for doc, score in results if score >= threshold]
    logger.debug(
        f"Retrieved {len(filtered)}/{k} chunks above threshold {threshold:.2f}"
        + (f" with filter" if where else "")
    )
    return filtered
