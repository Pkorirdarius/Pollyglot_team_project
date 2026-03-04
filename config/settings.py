"""
config/settings.py
──────────────────
Centralised settings loaded from .env via Pydantic-Settings.

Token-budget defaults are conservative to minimise API costs:
  • LLM max_tokens      : 1024  (was 2048 — most answers need far less)
  • chunk_size          : 400   (was 512  — smaller chunks = fewer tokens in context)
  • top_k_retrieval     : 3     (was 5    — send 3 best chunks, not 5)
  • similarity_threshold: 0.75  (was 0.72 — higher bar = less noise in context)
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM Provider ──────────────────────────────────────────────────────────
    gemini_api_key:    str = Field("", alias="GEMINI_API_KEY")
    anthropic_api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    openai_api_key:    str = Field("", alias="OPENAI_API_KEY")

    llm_provider: Literal["gemini", "anthropic", "openai"] = Field(
        "gemini", alias="LLM_PROVIDER"
    )

    # FIX 1: gemini-2.0-flash hits free-tier quota fast.
    #         gemini-1.5-flash has a much higher free-tier limit (1500 RPD).
    #         Switch back to gemini-2.0-flash once you add billing.
    llm_model: str = Field("gemini-1.5-flash", alias="LLM_MODEL")

    # FIX 2: The correct embedding model name for the new google-genai SDK is
    #         "text-embedding-004" (no "models/" prefix — the SDK adds it).
    #         "models/text-embedding-004" causes a 404 on v1beta.
    embedding_model: str = Field("text-embedding-004", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(768, alias="EMBEDDING_DIMENSION")

    # TOKEN OPTIMISATION: 1024 is enough for translation answers.
    # The LLM will rarely need more; cutting this halves worst-case token spend.
    max_tokens:  int   = Field(1024,  alias="MAX_TOKENS")
    temperature: float = Field(0.2,   alias="TEMPERATURE")

    # ── Vector Store ──────────────────────────────────────────────────────────
    vectorstore_provider: Literal["chroma", "pinecone"] = Field(
        "chroma", alias="VECTORSTORE_PROVIDER"
    )
    chroma_persist_dir:  Path = Field(Path("./data/vectorstore"), alias="CHROMA_PERSIST_DIR")
    pinecone_api_key:    str  = Field("", alias="PINECONE_API_KEY")
    pinecone_index_name: str  = Field("rag-index", alias="PINECONE_INDEX_NAME")
    pinecone_env:        str  = Field("us-east-1-aws", alias="PINECONE_ENV")

    # ── RAG Pipeline — TOKEN OPTIMISATION ────────────────────────────────────
    # Smaller chunks → fewer tokens per retrieved chunk in the LLM prompt.
    chunk_size:    int = Field(400,  alias="CHUNK_SIZE")
    chunk_overlap: int = Field(40,   alias="CHUNK_OVERLAP")

    # Fewer chunks retrieved → smaller prompt → lower cost.
    # 3 high-quality chunks beats 5 noisy ones for translation tasks.
    top_k_retrieval:    int   = Field(3,    alias="TOP_K_RETRIEVAL")
    similarity_threshold: float = Field(0.75, alias="SIMILARITY_THRESHOLD")

    # ── Data Paths ────────────────────────────────────────────────────────────
    raw_data_dir:       Path = Field(Path("./data/raw"),       alias="RAW_DATA_DIR")
    processed_data_dir: Path = Field(Path("./data/processed"), alias="PROCESSED_DATA_DIR")

    # ── API ───────────────────────────────────────────────────────────────────
    api_host:   str  = Field("0.0.0.0", alias="API_HOST")
    api_port:   int  = Field(8000,      alias="API_PORT")
    api_reload: bool = Field(True,      alias="API_RELOAD")
    log_level:  str  = Field("info",    alias="LOG_LEVEL")

    # ── UI ────────────────────────────────────────────────────────────────────
    ui_framework: Literal["streamlit", "gradio"] = Field(
        "streamlit", alias="UI_FRAMEWORK"
    )


# Singleton — import this everywhere
settings = Settings()
