"""
config/settings.py
─────────────────
Centralised settings loaded from the .env file via Pydantic-Settings.
All other modules import from here — never read os.environ directly.
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
    gemini_api_key: str = Field("", alias="GEMINI_API_KEY")
    anthropic_api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    
    # Updated to include "gemini"
    llm_provider: Literal["gemini", "anthropic", "openai"] = Field(
        "gemini", alias="LLM_PROVIDER"
    )
    
    # Defaulting to 2026 flagship models
    llm_model: str = Field("gemini-2.0-flash", alias="LLM_MODEL")
    embedding_model: str = Field("models/text-embedding-004", alias="EMBEDDING_MODEL")
    
    # Note: text-embedding-004 is 768, OpenAI small is 1536. 
    # Ensure this matches your existing index if you aren't starting fresh!
    embedding_dimension: int = Field(768, alias="EMBEDDING_DIMENSION")
    
    max_tokens: int = Field(2048, alias="MAX_TOKENS")
    temperature: float = Field(0.2, alias="TEMPERATURE")

    # ── Vector Store ──────────────────────────────────────────────────────────
    vectorstore_provider: Literal["chroma", "pinecone"] = Field(
        "chroma", alias="VECTORSTORE_PROVIDER"
    )
    chroma_persist_dir: Path = Field(Path("./data/vectorstore"), alias="CHROMA_PERSIST_DIR")
    pinecone_api_key: str = Field("", alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field("rag-index", alias="PINECONE_INDEX_NAME")
    pinecone_env: str = Field("us-east-1-aws", alias="PINECONE_ENV")

    # ── RAG Pipeline ──────────────────────────────────────────────────────────
    chunk_size: int = Field(512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(64, alias="CHUNK_OVERLAP")
    top_k_retrieval: int = Field(5, alias="TOP_K_RETRIEVAL")
    similarity_threshold: float = Field(0.72, alias="SIMILARITY_THRESHOLD")

    # ── Data Paths ────────────────────────────────────────────────────────────
    raw_data_dir: Path = Field(Path("./data/raw"), alias="RAW_DATA_DIR")
    processed_data_dir: Path = Field(Path("./data/processed"), alias="PROCESSED_DATA_DIR")

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")
    api_reload: bool = Field(True, alias="API_RELOAD")
    log_level: str = Field("info", alias="LOG_LEVEL")

    # ── UI ────────────────────────────────────────────────────────────────────
    ui_framework: Literal["streamlit", "gradio"] = Field("streamlit", alias="UI_FRAMEWORK")


# Singleton – import this everywhere
settings = Settings()