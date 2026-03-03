"""
data_wrangling/vectorstore.py
Build, persist, and query a vector store.
Supports: ChromaDB (local) and Pinecone (cloud).
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from langchain.schema import Document
from loguru import logger
from config.settings import settings

if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStore

def _get_embeddings():
    """Returns embeddings based on the provider set in settings."""
    if settings.llm_provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model, # e.g., "models/text-embedding-004"
            google_api_key=settings.gemini_api_key,
            task_type="retrieval_document" # Optimized for indexing
        )
    
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )

def get_vectorstore(collection_name: str = "rag_store") -> "VectorStore":
    embeddings = _get_embeddings()
    if settings.vectorstore_provider == "chroma":
        from langchain_community.vectorstores import Chroma
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(settings.chroma_persist_dir),
        )
    raise ValueError(f"Unknown vectorstore provider: {settings.vectorstore_provider}")

def ingest_documents(documents: list[Document], collection_name: str = "rag_store") -> int:
    if not documents:
        logger.warning("ingest_documents called with empty list.")
        return 0
    
    embeddings = _get_embeddings()
    # Explicitly ensure we are using the document task type for ingestion
    if hasattr(embeddings, "task_type"):
        embeddings.task_type = "retrieval_document"

    if settings.vectorstore_provider == "chroma":
        from langchain_community.vectorstores import Chroma
        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=str(settings.chroma_persist_dir),
        )
    elif settings.vectorstore_provider == "pinecone":
        from langchain_pinecone import PineconeVectorStore
        PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=settings.pinecone_index_name,
        )
    
    logger.success(f"Ingested {len(documents)} chunks into {collection_name!r}")
    return len(documents)

def similarity_search(
    query: str,
    top_k: int | None = None,
    collection_name: str = "rag_store",
    score_threshold: float | None = None,
) -> list[tuple[Document, float]]:
    k = top_k or settings.top_k_retrieval
    threshold = score_threshold or settings.similarity_threshold
    
    # Update task_type for the query phase
    embeddings = _get_embeddings()
    if hasattr(embeddings, "task_type"):
        embeddings.task_type = "retrieval_query"

    vs = get_vectorstore(collection_name)
    results = vs.similarity_search_with_relevance_scores(query, k=k)
    
    filtered = [(doc, score) for doc, score in results if score >= threshold]
    logger.debug(f"Retrieved {len(filtered)}/{k} chunks above threshold {threshold:.2f}")
    return filtered