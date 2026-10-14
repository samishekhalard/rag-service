"""
Configuration precedence.

The documented order is: explicit arguments > environment > default.yaml >
field defaults. It was wrong in both directions at once — YAML beat the
environment, because every YAML value was passed as a constructor argument and
pydantic-settings ranks those above env vars. docker-compose sets
RAG_QDRANT_URL to reach the qdrant container, and that was being discarded.
"""

import pytest

from rag.config import load_settings


def test_environment_overrides_yaml(monkeypatch):
    """The regression: default.yaml holds qdrant_url, so the env var lost."""
    monkeypatch.setenv("RAG_QDRANT_URL", ":memory:")
    assert load_settings().qdrant_url == ":memory:"


def test_explicit_argument_beats_environment(monkeypatch):
    monkeypatch.setenv("RAG_QDRANT_URL", ":memory:")
    assert load_settings(qdrant_url="http://elsewhere:6333").qdrant_url == "http://elsewhere:6333"


def test_yaml_applies_when_environment_is_silent(monkeypatch):
    monkeypatch.delenv("RAG_QDRANT_URL", raising=False)
    assert load_settings().qdrant_url == "http://localhost:6333"


@pytest.mark.parametrize(
    "variable,attribute,value",
    [
        ("RAG_PROVIDER", "provider", "local"),
        ("RAG_CHUNK_SIZE", "chunk_size", 512),
        ("RAG_RERANK", "rerank", False),
        ("RAG_OLLAMA_CHAT_MODEL", "ollama_chat_model", "qwen2.5:7b"),
    ],
)
def test_every_yaml_key_is_overridable_from_the_environment(
    monkeypatch, variable, attribute, value
):
    monkeypatch.setenv(variable, str(value))
    assert getattr(load_settings(), attribute) == value
