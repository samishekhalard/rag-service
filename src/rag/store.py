"""
Qdrant-backed vector store.

Accepts either a server URL or ":memory:". The in-memory mode runs Qdrant
inside the process, which is what the test suite uses — the same code path
as production, without a container.

Chunk ids are 32-char blake2b digests, which map exactly onto the UUID form
Qdrant requires for string point ids, so ingestion is idempotent: re-ingesting
an unchanged document overwrites the same points instead of duplicating them.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from .chunking import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float

    @property
    def id(self) -> str:
        return self.chunk.id


def _point_id(chunk_id: str) -> str:
    return str(uuid.UUID(hex=chunk_id))


class VectorStore:
    def __init__(self, url: str, collection: str, dimension: int) -> None:
        self.collection = collection
        self.dimension = dimension
        if url == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(url=url)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing:
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.dimension, distance=models.Distance.COSINE
            ),
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return 0
        points = [
            models.PointStruct(
                id=_point_id(chunk.id),
                vector=vector,
                payload={
                    "chunk_id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "ordinal": chunk.ordinal,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    def search(self, vector: list[float], top_k: int) -> list[ScoredChunk]:
        hits = self.client.query_points(
            collection_name=self.collection, query=vector, limit=top_k, with_payload=True
        ).points
        return [
            ScoredChunk(
                chunk=Chunk(
                    id=hit.payload["chunk_id"],
                    doc_id=hit.payload["doc_id"],
                    ordinal=hit.payload["ordinal"],
                    text=hit.payload["text"],
                    metadata=hit.payload.get("metadata") or {},
                ),
                score=float(hit.score),
            )
            for hit in hits
        ]

    def all_chunks(self) -> list[Chunk]:
        """Every stored chunk. The BM25 index is built from this."""
        chunks: list[Chunk] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            chunks.extend(
                Chunk(
                    id=p.payload["chunk_id"],
                    doc_id=p.payload["doc_id"],
                    ordinal=p.payload["ordinal"],
                    text=p.payload["text"],
                    metadata=p.payload.get("metadata") or {},
                )
                for p in points
            )
            if offset is None:
                break
        return chunks

    def get(self, chunk_id: str) -> Chunk | None:
        """Exact lookup by chunk id. Returns None when the id is unknown."""
        try:
            point_id = _point_id(chunk_id)
        except ValueError:
            # Not a 32-char hex digest, so it cannot be one of our ids.
            return None
        points = self.client.retrieve(
            collection_name=self.collection, ids=[point_id], with_payload=True
        )
        if not points:
            return None
        payload = points[0].payload
        return Chunk(
            id=payload["chunk_id"],
            doc_id=payload["doc_id"],
            ordinal=payload["ordinal"],
            text=payload["text"],
            metadata=payload.get("metadata") or {},
        )

    def by_document(self, doc_id: str) -> list[Chunk]:
        """Every chunk of one document, in reading order."""
        chunks = [c for c in self.all_chunks() if c.doc_id == doc_id]
        chunks.sort(key=lambda c: c.ordinal)
        return chunks

    def count(self) -> int:
        return int(self.client.count(collection_name=self.collection, exact=True).count)

    def drop(self) -> None:
        self.client.delete_collection(collection_name=self.collection)
        self._ensure_collection()
