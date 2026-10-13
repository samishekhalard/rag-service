"""
Evaluation metrics.

Retrieval ground truth is recorded at document level, not chunk level. Chunk
ids change whenever chunk_size changes, so chunk-level ground truth would
silently invalidate the eval set on a config change; document ids survive
re-chunking, which keeps results comparable across configurations.

Answer metrics are deliberately cheap and transparent — term coverage and
groundedness, not an LLM judge. An LLM judge scored by the same model family
that generated the answer measures agreement, not correctness, and it cannot
be reproduced by someone cloning the repo without keys.
"""

from __future__ import annotations

import math
import re

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def dedupe(docs: list[str]) -> list[str]:
    """
    Collapse a chunk-level ranking to a document-level one, keeping the best
    rank each document achieved.

    Retrieval returns several chunks per document, so the raw list repeats
    document ids. Scoring that list directly double-counts a single relevant
    document at several ranks and pushes nDCG above 1.0, which is how you
    notice the bug. Ground truth is document-level, so the ranking must be too.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for doc in docs:
        if doc not in seen:
            seen.add(doc)
            ordered.append(doc)
    return ordered


def recall_at_k(retrieved_docs: list[str], relevant: set[str], k: int) -> float:
    """Share of relevant documents appearing in the top k."""
    if not relevant:
        return 0.0
    top = set(retrieved_docs[:k])
    return len(top & relevant) / len(relevant)


def hit_at_k(retrieved_docs: list[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant document is in the top k. The user-facing question."""
    return 1.0 if set(retrieved_docs[:k]) & relevant else 0.0


def precision_at_k(retrieved_docs: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved_docs[:k]
    if not top:
        return 0.0
    return sum(1 for doc in top if doc in relevant) / len(top)


def reciprocal_rank(retrieved_docs: list[str], relevant: set[str]) -> float:
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_docs: list[str], relevant: set[str], k: int) -> float:
    """Binary-gain nDCG. Rewards putting relevant documents early."""
    if not relevant:
        return 0.0
    gains = [1.0 if doc in relevant else 0.0 for doc in retrieved_docs[:k]]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def keyword_coverage(answer: str, must_include: list[str]) -> float:
    """Share of required strings present in the answer, case-insensitive."""
    if not must_include:
        return 1.0
    lowered = answer.lower()
    return sum(1 for term in must_include if term.lower() in lowered) / len(must_include)


def groundedness(answer: str, context: str) -> float:
    """
    Share of the answer's content words that also appear in the retrieved
    context. Not a hallucination detector — a low score means the answer
    introduced vocabulary the context never supplied, which is where to look.
    """
    answer_tokens = {t for t in _tokens(answer) if len(t) > 3}
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & _tokens(context)) / len(answer_tokens)


def refusal(answer: str) -> bool:
    """Detect a refusal, for cases whose answer is deliberately absent."""
    markers = (
        "does not contain", "not contain", "no information", "cannot answer",
        "can't answer", "not in the context", "no relevant context",
        "unable to answer", "не содержит", "нет информации", "не могу ответить",
    )
    lowered = answer.lower()
    return any(marker in lowered for marker in markers)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile. p50 and p95 of latency are reported from this."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q / 100 * len(ordered)) - 1))
    return ordered[index]
