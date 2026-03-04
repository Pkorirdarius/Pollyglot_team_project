"""
data_wrangling/loader.py
Updated for 2026: Enhanced PDF page tracking and modern web scraping.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredURLLoader
from loguru import logger

from config.settings import settings

if TYPE_CHECKING:
    pass


# ── Splitter (shared) ─────────────────────────────────────────────────────────

def _make_splitter() -> RecursiveCharacterTextSplitter:
    # 2026 Tip: With Gemini's 1M+ context window, you can actually increase 
    # CHUNK_SIZE to 1024 or 2048 if you want more coherent retrieval.
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# ── Individual loaders ────────────────────────────────────────────────────────

def _load_pdf(path: Path) -> list[Document]:
    # PyMuPDFLoader is generally faster and better at preserving page numbers 
    # than the standard PyPDFLoader in recent LangChain versions.
    
    loader = PyMuPDFLoader(str(path))
    return loader.load()


def _load_txt(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    # Store filename without full system path for cleaner UI display
    return [Document(page_content=text, metadata={"source": path.name})]


def _load_docx(path: Path) -> list[Document]:
    loader = Docx2txtLoader(str(path))
    return loader.load()


def _load_csv(path: Path) -> list[Document]:
    docs: list[Document] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Formatted string for better vector representation
            text = " | ".join(f"{k.strip()}: {v.strip()}" for k, v in row.items() if v)
            docs.append(Document(
                page_content=text, 
                metadata={"source": path.name, "row": i}
            ))
    return docs


def _load_url(url: str) -> list[Document]:
    # 2026 Standard: WebBaseLoader often fails on React/Next.js sites.
    # Using 'UnstructuredURLLoader' or 'Playwright' is preferred.
    loader = UnstructuredURLLoader(urls=[url])
    return loader.load()


# ── Public API ────────────────────────────────────────────────────────────────

def load_and_split(
    source: str,
    source_type: str = "pdf",
    extra_metadata: dict | None = None,
) -> list[Document]:
    """
    Load a document and split into chunks with enriched metadata.
    """
    extra_metadata = extra_metadata or {}
    path = Path(source) if source_type != "url" else None
    logger.info(f"Loading [{source_type}] → {source}")

    _LOADERS = {
        "pdf":  lambda: _load_pdf(path),
        "txt":  lambda: _load_txt(path),
        "docx": lambda: _load_docx(path),
        "csv":  lambda: _load_csv(path),
        "url":  lambda: _load_url(source),
    }

    if source_type not in _LOADERS:
        raise ValueError(f"Unsupported source_type '{source_type}'.")

    try:
        raw_docs = _LOADERS[source_type]()
    except Exception as e:
        logger.error(f"Failed to load {source}: {e}")
        raise

    # 1. Clean and enrich metadata before splitting
    for doc in raw_docs:
        # Ensure source is just the filename, not the full temp path
        if "source" in doc.metadata and source_type != "url":
            doc.metadata["source"] = Path(doc.metadata["source"]).name
        
        doc.metadata.update(extra_metadata)

    # 2. Split into chunks
    splitter = _make_splitter()
    chunks = splitter.split_documents(raw_docs)

    # 3. Final polish: Ensure every chunk knows its source for the UI
    for chunk in chunks:
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = source

    logger.success(f"Processed {source}: {len(raw_docs)} pages → {len(chunks)} chunks")
    return chunks