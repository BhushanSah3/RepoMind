"""Centralized, environment-driven settings for RepoMind."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    default_llm_provider: Literal["gemini", "groq", "ollama"] = "gemini"
    ollama_base_url: str = "http://localhost:11434"

    gemini_model: str = "gemini-3.6-flash"
    groq_model: str = "qwen/qwen3.8-27b"
    ollama_model: str = "llama3.1"

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_batch_size: int = 64
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    data_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data")
    chroma_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "chroma")
    graph_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "graphs")

    max_file_size_bytes: int = 1_000_000
    retrieval_top_k: int = 10
    dense_retrieval_top_k: int = 20
    rerank_top_k: int = 8
    rrf_k: int = 60
    rag_faithfulness_threshold: float = 0.7
    rag_context_relevance_threshold: float = 0.7
    rag_answer_relevance_threshold: float = 0.7
    max_retrieval_retries: int = 2


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings instance for the current process."""

    return Settings()
