"""
Reranking stage.

Fusion gives a wide candidate list ordered by rank agreement; the reranker
reorders the top of it by actually reading query and passage together. This
is the step that most "RAG in 50 lines" write-ups omit, and it is usually
worth more than swapping the embedding model.

Two implementations, both honest about what they do:

- LLMReranker scores each candidate with the chat model in a single call.
  It costs tokens and latency, which the eval table reports.
- LexicalReranker scores by query-term coverage and proximity. No network,
  no tokens. It is the default when the provider is "fake", and a reasonable
  floor to compare the LLM reranker against — if the LLM reranker cannot beat
  term coverage, it is not earning its latency.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from .config import Settings
from .retrieval import FusedChunk, tokenize


@dataclass
class RerankResult:
    chunks: list[FusedChunk]
    tokens_in: int = 0
    tokens_out: int = 0


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[FusedChunk], top_n: int) -> RerankResult: ...


class IdentityReranker:
    """Keeps fusion order. Used to measure what reranking is actually worth."""

    name = "identity"

    def rerank(self, query: str, candidates: list[FusedChunk], top_n: int) -> RerankResult:
        return RerankResult(candidates[:top_n])


class LexicalReranker:
    """
    Query-term coverage with a proximity bonus.

    coverage = share of distinct query terms present in the passage
    proximity = 1 / (1 + smallest window containing the most terms), scaled

    Deterministic and offline, which makes it testable and makes it a fair
    baseline rather than a placeholder.
    """

    name = "lexical"

    def _score(self, query_terms: set[str], text: str) -> float:
        tokens = tokenize(text)
        if not tokens or not query_terms:
            return 0.0
        positions = [i for i, token in enumerate(tokens) if token in query_terms]
        if not positions:
            return 0.0
        present = {tokens[i] for i in positions}
        coverage = len(present) / len(query_terms)
        span = positions[-1] - positions[0] + 1
        proximity = len(present) / span if span else 1.0
        return 0.75 * coverage + 0.25 * proximity

    def rerank(self, query: str, candidates: list[FusedChunk], top_n: int) -> RerankResult:
        query_terms = set(tokenize(query))
        scored = [
            FusedChunk(
                chunk=c.chunk,
                score=self._score(query_terms, c.chunk.text),
                dense_rank=c.dense_rank,
                sparse_rank=c.sparse_rank,
            )
            for c in candidates
        ]
        # Stable on ties: fusion order is the tiebreaker, not dict ordering.
        order = {c.id: i for i, c in enumerate(candidates)}
        scored.sort(key=lambda c: (-c.score, order[c.id]))
        return RerankResult(scored[:top_n])


_SCORE_LINE = re.compile(r"(\d+)\s*[:.)-]\s*([0-9]*\.?[0-9]+)")

_PROMPT = """You score how well each passage answers the question.

Question: {query}

Passages:
{passages}

Return JSON only: a list of objects with "id" (the passage number) and
"score" (0.0 to 1.0). No prose, no code fence."""


class LLMReranker:
    """Single-call listwise reranker. Falls back to fusion order on a bad reply."""

    name = "llm"

    def __init__(self, chat, settings: Settings) -> None:
        self.chat = chat
        self.settings = settings

    def rerank(self, query: str, candidates: list[FusedChunk], top_n: int) -> RerankResult:
        if not candidates:
            return RerankResult([])

        budget = max(200, self.settings.max_context_chars // max(1, len(candidates)))
        passages = "\n\n".join(
            f"[{i}] {c.chunk.text[:budget]}" for i, c in enumerate(candidates)
        )
        reply = self.chat.complete(
            system="You are a precise relevance scorer. Output JSON only.",
            user=_PROMPT.format(query=query, passages=passages),
        )

        scores = self._parse(reply.text, len(candidates))
        if scores is None:
            return RerankResult(candidates[:top_n], reply.tokens_in, reply.tokens_out)

        order = {c.id: i for i, c in enumerate(candidates)}
        rescored = [
            FusedChunk(
                chunk=c.chunk,
                score=scores[i],
                dense_rank=c.dense_rank,
                sparse_rank=c.sparse_rank,
            )
            for i, c in enumerate(candidates)
        ]
        rescored.sort(key=lambda c: (-c.score, order[c.id]))
        return RerankResult(rescored[:top_n], reply.tokens_in, reply.tokens_out)

    @staticmethod
    def _parse(text: str, count: int) -> list[float] | None:
        """Tolerate fenced JSON and 'index: score' lines; give up cleanly."""
        cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        scores = [0.0] * count
        try:
            payload = json.loads(cleaned)
            for item in payload:
                index = int(item["id"])
                if 0 <= index < count:
                    scores[index] = float(item["score"])
            return scores
        except Exception:
            pass

        found = _SCORE_LINE.findall(cleaned)
        if not found:
            return None
        for raw_index, raw_score in found:
            index = int(raw_index)
            if 0 <= index < count:
                scores[index] = float(raw_score)
        return scores


def build_reranker(settings: Settings, chat=None) -> Reranker:
    if not settings.rerank:
        return IdentityReranker()
    if settings.provider == "fake" or chat is None:
        return LexicalReranker()
    return LLMReranker(chat, settings)
