"""
Hybrid retrieval: dense vectors + BM25, combined with reciprocal rank fusion.

Why RRF rather than a weighted score blend: cosine similarity and BM25 live
on incompatible scales, and normalising them per query makes the blend depend
on how good the best hit happens to be. RRF only uses rank, so it needs no
tuning and no calibration, which matters when the embedding model is a config
switch away from changing.

    score(d) = sum over lists of 1 / (k + rank(d))

k dampens the influence of top positions; 60 is the value from the original
Cormack et al. paper and the default here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .chunking import Chunk
from .store import ScoredChunk

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class FusedChunk:
    chunk: Chunk
    score: float
    dense_rank: int | None
    sparse_rank: int | None

    @property
    def id(self) -> str:
        return self.chunk.id


class BM25Index:
    """Lexical half of the hybrid. Rebuilt from the store after ingestion."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._corpus = [tokenize(c.text) for c in chunks]
        # BM25Okapi rejects an empty corpus; keep the index inert instead.
        self._bm25 = BM25Okapi(self._corpus) if self._corpus else None

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        if self._bm25 is None or top_k <= 0:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)[:top_k]
        return [
            ScoredChunk(chunk=self.chunks[i], score=float(score))
            for i, score in ranked
            if score > 0.0
        ]


def reciprocal_rank_fusion(
    dense: list[ScoredChunk],
    sparse: list[ScoredChunk],
    k: int = 60,
    limit: int | None = None,
) -> list[FusedChunk]:
    """Merge two ranked lists. Ties break on dense rank, then chunk id."""
    dense_ranks = {hit.id: rank for rank, hit in enumerate(dense, start=1)}
    sparse_ranks = {hit.id: rank for rank, hit in enumerate(sparse, start=1)}
    by_id: dict[str, Chunk] = {hit.id: hit.chunk for hit in list(dense) + list(sparse)}

    fused: list[FusedChunk] = []
    for chunk_id, chunk in by_id.items():
        dense_rank = dense_ranks.get(chunk_id)
        sparse_rank = sparse_ranks.get(chunk_id)
        score = 0.0
        if dense_rank is not None:
            score += 1.0 / (k + dense_rank)
        if sparse_rank is not None:
            score += 1.0 / (k + sparse_rank)
        fused.append(
            FusedChunk(
                chunk=chunk, score=score, dense_rank=dense_rank, sparse_rank=sparse_rank
            )
        )

    fused.sort(
        key=lambda f: (-f.score, f.dense_rank if f.dense_rank is not None else 10**6, f.id)
    )
    return fused[:limit] if limit else fused
