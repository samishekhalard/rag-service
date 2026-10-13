"""
Document chunking.

Splits on paragraph boundaries first and only falls back to hard character
cuts when a single paragraph exceeds the budget. Chunks carry their source
document id and ordinal, which is what the eval harness matches against to
compute recall — retrieval metrics need stable chunk identity, so ids are
derived from content, not from insertion order.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_PARAGRAPH = re.compile(r"\n\s*\n")
_WHITESPACE = re.compile(r"[ \t]+")


@dataclass
class Chunk:
    id: str
    doc_id: str
    ordinal: int
    text: str
    metadata: dict = field(default_factory=dict)


def _clean(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _split_oversized(paragraph: str, size: int, overlap: int) -> list[str]:
    """Hard-cut a paragraph that cannot fit, keeping `overlap` chars of tail."""
    pieces: list[str] = []
    start = 0
    step = max(1, size - overlap)
    while start < len(paragraph):
        pieces.append(paragraph[start : start + size])
        if start + size >= len(paragraph):
            break
        start += step
    return pieces


def chunk_text(
    text: str,
    doc_id: str,
    size: int = 800,
    overlap: int = 120,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Chunk one document. Returns [] for empty or whitespace-only input."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    paragraphs = [_clean(p) for p in _PARAGRAPH.split(text)]
    paragraphs = [p for p in paragraphs if p]
    if not paragraphs:
        return []

    buffers: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > size:
            if current:
                buffers.append(current)
                current = ""
            buffers.extend(_split_oversized(paragraph, size, overlap))
            continue
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= size:
            current = candidate
        else:
            buffers.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{paragraph}".strip() if tail else paragraph
    if current:
        buffers.append(current)

    chunks: list[Chunk] = []
    for ordinal, body in enumerate(buffers):
        digest = hashlib.blake2b(
            f"{doc_id}:{ordinal}:{body}".encode("utf-8"), digest_size=16
        ).hexdigest()
        chunks.append(
            Chunk(
                id=digest,
                doc_id=doc_id,
                ordinal=ordinal,
                text=body,
                metadata=dict(metadata or {}),
            )
        )
    return chunks
