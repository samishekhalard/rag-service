"""
Configuration for the RAG service.

One switch decides where embeddings and generation happen: `provider`.
Set it to "cloud" (OpenAI-compatible HTTP API) or "local" (Ollama). Nothing
else in the codebase branches on the provider — everything downstream talks
to the Embedder and ChatModel protocols in embeddings.py and llm.py.

Settings come from, in order of precedence:
    1. explicit arguments
    2. environment variables (RAG_ prefix) / .env
    3. config/default.yaml
    4. the defaults below
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["cloud", "local", "fake"]
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


class Settings(BaseSettings):
    """Runtime configuration. Every value is overridable by RAG_* env vars."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- provider switch ------------------------------------------------
    # "fake" is for tests: deterministic embeddings, no network, no keys.
    provider: Provider = "cloud"

    cloud_base_url: str = "https://api.openai.com/v1"
    cloud_api_key: str = ""
    cloud_embed_model: str = "text-embedding-3-small"
    cloud_chat_model: str = "gpt-4o-mini"

    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_chat_model: str = "llama3.1:8b"

    # --- vector store ---------------------------------------------------
    # ":memory:" runs Qdrant in-process — used by the test suite so the
    # suite needs no running container.
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"

    # --- retrieval ------------------------------------------------------
    chunk_size: int = 800
    chunk_overlap: int = 120
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 60
    rerank_top_n: int = 5
    rerank: bool = True

    # --- tracing --------------------------------------------------------
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # --- pricing, USD per 1M tokens -------------------------------------
    # Used to report cost per query. Local models are free at the API
    # level; the eval table reports 0.0 for them, which is the honest
    # number for API spend and says nothing about hardware cost.
    price_embed_per_mtok: float = 0.02
    price_input_per_mtok: float = 0.15
    price_output_per_mtok: float = 0.60

    max_context_chars: int = 6000
    request_timeout_s: float = 60.0

    extra: dict = Field(default_factory=dict)

    @property
    def embed_model(self) -> str:
        return self.cloud_embed_model if self.provider == "cloud" else self.ollama_embed_model

    @property
    def chat_model(self) -> str:
        return self.cloud_chat_model if self.provider == "cloud" else self.ollama_chat_model

    @property
    def variant(self) -> str:
        """Short label for this configuration, used as the eval row key."""
        if self.provider == "fake":
            return "fake"
        suffix = "+rerank" if self.rerank else "-rerank"
        return f"{self.provider}:{self.chat_model}{suffix}"


def load_settings(path: Path | None = None, **overrides) -> Settings:
    """Load YAML defaults, then let env vars and explicit overrides win."""
    base: dict = {}
    config_path = path or CONFIG_PATH
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        base = {k: v for k, v in loaded.items() if v is not None}
    base.update(overrides)
    return Settings(**base)
