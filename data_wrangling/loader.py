"""
data_wrangling/loader.py
─────────────────────────
Document loading + splitting with rich DocumentMetadata tagging.

Supports: PDF, TXT, DOCX, CSV, URL
Auto-detects document language when langdetect is installed.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    UnstructuredURLLoader,
)
from loguru import logger

from config.settings import settings
from search.models.schemas import DocumentMetadata

if TYPE_CHECKING:
    pass


# ── Language auto-detection (optional dependency) ─────────────────────────────

def _detect_language(text: str) -> str | None:
    try:
        from langdetect import detect
        return detect(text[:2000])          # sample first 2000 chars for speed
    except Exception:
        return None


# ── Splitter ──────────────────────────────────────────────────────────────────

def _make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# ── Individual loaders ────────────────────────────────────────────────────────

def _load_pdf(path: Path) -> list[Document]:
    loader = PyMuPDFLoader(str(path))
    docs = loader.load()
    # Normalise page numbers to 1-based integers
    total = len(docs)
    for i, doc in enumerate(docs):
        doc.metadata["page"] = int(doc.metadata.get("page", i)) + 1
        doc.metadata["total_pages"] = total
    return docs


def _load_txt(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": path.name})]


def _load_docx(path: Path) -> list[Document]:
    loader = Docx2txtLoader(str(path))
    return loader.load()


def _load_csv(path: Path) -> list[Document]:
    docs: list[Document] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = " | ".join(
                f"{k.strip()}: {v.strip()}" for k, v in row.items() if v
            )
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": path.name, "row": i},
                )
            )
    return docs


def _load_url(url: str) -> list[Document]:
    loader = UnstructuredURLLoader(urls=[url])
    return loader.load()


# ── Public API ────────────────────────────────────────────────────────────────

def load_and_split(
    source: str,
    source_type: str = "pdf",
    extra_metadata: dict | None = None,
    source_language: str | None = None,
    target_language: str | None = None,
    domain: str = "general",
    register: str = "neutral",
) -> list[Document]:
    """
    Load a document, attach rich DocumentMetadata, and split into chunks.

    Args:
        source:          File path or URL.
        source_type:     One of: pdf | txt | docx | csv | url.
        extra_metadata:  Arbitrary key-value pairs merged into metadata.
        source_language: BCP-47 code for the document's language (auto-detected if omitted).
        target_language: BCP-47 code for the translation target language.
        domain:          Subject domain (general / legal / medical / …).
        register:        Linguistic register (formal / informal / neutral).

    Returns:
        List of LangChain Document chunks with enriched metadata.
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

    # ── Enrich metadata ───────────────────────────────────────────────────────
    # Detect language from first document's content if not supplied
    detected_lang = source_language
    if not detected_lang and raw_docs:
        detected_lang = _detect_language(raw_docs[0].page_content)
        if detected_lang:
            logger.info(f"Auto-detected language: {detected_lang}")

    source_filename = (
        Path(source).name if source_type != "url" else source
    )
    total_pages = len(raw_docs) if source_type == "pdf" else None

    for i, doc in enumerate(raw_docs):
        doc_meta = DocumentMetadata(
            source=source_filename,
            chunk_index=i,
            page=doc.metadata.get("page"),
            total_pages=total_pages,
            source_language=detected_lang,
            target_language=target_language,
            language_pair=(
                f"{detected_lang}-{target_language}"
                if detected_lang and target_language
                else None
            ),
            domain=domain,          # type: ignore[arg-type]
            register=register,      # type: ignore[arg-type]
            doc_type=source_type,   # type: ignore[arg-type]
            extra=extra_metadata,
        )
        # Merge: existing loader metadata wins for page/source; ours fills the rest
        doc.metadata = {**doc_meta.to_chroma_dict(), **doc.metadata}
        # Always normalise source to filename only
        doc.metadata["source"] = source_filename

    # ── Split ─────────────────────────────────────────────────────────────────
    splitter = _make_splitter()
    chunks = splitter.split_documents(raw_docs)

    # ── Final polish ──────────────────────────────────────────────────────────
    for idx, chunk in enumerate(chunks):
        chunk.metadata.setdefault("source", source_filename)
        chunk.metadata["chunk_index"] = idx      # reindex after splitting

    logger.success(
        f"Processed {source_filename}: "
        f"{len(raw_docs)} pages → {len(chunks)} chunks "
        f"[lang={detected_lang}, domain={domain}]"
    )
    return chunks
