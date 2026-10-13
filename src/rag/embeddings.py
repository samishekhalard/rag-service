"""
Embedding backends behind one protocol.

Three implementations: an OpenAI-compatible HTTP API, Ollama, and a
deterministic fake used by the test suite. The fake is not a mock object —
it produces stable, well-spread unit vectors from token hashes, so retrieval
behaviour is genuinely exercised in tests without a model or a network.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol

import httpx

from .config import Settings

_TOKEN = re.compile(r"\w+", re.UNICODE)


@dataclass
class EmbedResult:
    vectors: list[list[float]]
    tokens: int


class Embedder(Protocol):
    dimension: int

    def embed(self, texts: list[str]) -> EmbedResult: ...


def _estimate_tokens(texts: list[str]) -> int:
    """Rough token count for backends that don't report usage."""
    return max(1, sum(len(t) for t in texts) // 4)


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm < 1e-12:
        return vector
    return [v / norm for v in vector]


class FakeEmbedder:
    """
    Deterministic bag-of-words embedder. No network, no weights.

    Each token is hashed to a fixed dimension and accumulated, then the
    vector is L2-normalised. Documents sharing vocabulary land close
    together under cosine similarity, which is enough to test chunking,
    fusion, reranking and the end-to-end pipeline honestly.
    """

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return _normalise(vector)

    def embed(self, texts: list[str]) -> EmbedResult:
        return EmbedResult([self._one(t) for t in texts], _estimate_tokens(texts))


class CloudEmbedder:
    """OpenAI-compatible /embeddings endpoint."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.cloud_embed_model
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed(["dimension probe"]).vectors[0])
        return self._dimension

    def embed(self, texts: list[str]) -> EmbedResult:
        response = httpx.post(
            f"{self.settings.cloud_base_url.rstrip('/')}/embeddings",
            headers={"Authorization": f"Bearer {self.settings.cloud_api_key}"},
            json={"model": self.model, "input": texts},
            timeout=self.settings.request_timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        vectors = [item["embedding"] for item in sorted(payload["data"], key=lambda d: d["index"])]
        tokens = payload.get("usage", {}).get("total_tokens") or _estimate_tokens(texts)
        return EmbedResult(vectors, int(tokens))


class OllamaEmbedder:
    """Ollama /api/embed. Reports estimated tokens — Ollama does not bill."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.ollama_embed_model
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed(["dimension probe"]).vectors[0])
        return self._dimension

    def embed(self, texts: list[str]) -> EmbedResult:
        response = httpx.post(
            f"{self.settings.ollama_base_url.rstrip('/')}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=self.settings.request_timeout_s,
        )
        response.raise_for_status()
        vectors = response.json()["embeddings"]
        return EmbedResult(vectors, _estimate_tokens(texts))


def build_embedder(settings: Settings) -> Embedder:
    if settings.provider == "fake":
        return FakeEmbedder()
    if settings.provider == "local":
        return OllamaEmbedder(settings)
    if not settings.cloud_api_key:
        raise RuntimeError(
            "provider is 'cloud' but RAG_CLOUD_API_KEY is empty. "
            "Set the key, or use RAG_PROVIDER=local for Ollama."
        )
    return CloudEmbedder(settings)
