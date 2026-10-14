"""
The pipeline: ingest documents, answer questions.

Query path, each stage traced:

    embed query -> dense search  \
                                  >-- RRF fusion -> rerank -> generate
                   sparse search /

Retrieval is separable from generation on purpose: `retrieve()` returns the
ranked chunks and the trace, and the eval harness calls it directly to measure
recall without paying for generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .chunking import Chunk, chunk_text
from .config import Settings, load_settings
from .embeddings import build_embedder
from .llm import build_chat
from .rerank import build_reranker
from .retrieval import BM25Index, FusedChunk, reciprocal_rank_fusion
from .store import VectorStore
from .tracing import Trace, Tracer, cost_usd

SYSTEM_PROMPT = (
    "Answer the question using only the context provided. "
    "If the context does not contain the answer, say so plainly. "
    "Do not invent facts, citations or numbers."
)


@dataclass
class RetrievalOutcome:
    chunks: list[FusedChunk]
    trace: Trace
    embed_tokens: int = 0
    rerank_tokens_in: int = 0
    rerank_tokens_out: int = 0


@dataclass
class Answer:
    question: str
    text: str
    sources: list[dict] = field(default_factory=list)
    trace: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    latency_ms: float = 0.0


class RagPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.embedder = build_embedder(self.settings)
        self.chat = build_chat(self.settings)
        self.reranker = build_reranker(self.settings, self.chat)
        self.tracer = Tracer(self.settings)
        self.store = VectorStore(
            url=self.settings.qdrant_url,
            collection=self.settings.qdrant_collection,
            dimension=self.embedder.dimension,
        )
        self.bm25 = BM25Index(self.store.all_chunks())

    # ---------------------------------------------------------------- ingest

    def ingest_text(self, text: str, doc_id: str, metadata: dict | None = None) -> int:
        chunks = chunk_text(
            text,
            doc_id=doc_id,
            size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
            metadata=metadata,
        )
        return self._store(chunks)

    def ingest_path(self, path: str | Path, pattern: str = "*.md") -> int:
        path = Path(path)
        files = sorted(path.rglob(pattern)) if path.is_dir() else [path]
        chunks: list[Chunk] = []
        for file in files:
            chunks.extend(
                chunk_text(
                    file.read_text(encoding="utf-8"),
                    doc_id=file.stem,
                    size=self.settings.chunk_size,
                    overlap=self.settings.chunk_overlap,
                    metadata={"source": str(file)},
                )
            )
        return self._store(chunks)

    def _store(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = self.embedder.embed([c.text for c in chunks]).vectors
        written = self.store.upsert(chunks, vectors)
        # BM25 has no incremental update; rebuild from the store so the
        # lexical and dense halves always describe the same corpus.
        self.bm25 = BM25Index(self.store.all_chunks())
        return written

    # -------------------------------------------------------------- retrieve

    def retrieve(self, question: str, trace: Trace | None = None) -> RetrievalOutcome:
        trace = trace if trace is not None else Trace(name="retrieve")
        settings = self.settings

        with self.tracer.span(trace, "embed", model=settings.embed_model) as span:
            embedded = self.embedder.embed([question])
            span.tokens_in = embedded.tokens
        query_vector = embedded.vectors[0]

        with self.tracer.span(trace, "search_dense", top_k=settings.dense_top_k) as span:
            dense = self.store.search(query_vector, settings.dense_top_k)
            span.metadata["hits"] = len(dense)

        with self.tracer.span(trace, "search_sparse", top_k=settings.sparse_top_k) as span:
            sparse = self.bm25.search(question, settings.sparse_top_k)
            span.metadata["hits"] = len(sparse)

        with self.tracer.span(trace, "fuse", k=settings.rrf_k) as span:
            fused = reciprocal_rank_fusion(dense, sparse, k=settings.rrf_k)
            span.metadata["candidates"] = len(fused)
            span.metadata["overlap"] = sum(
                1 for f in fused if f.dense_rank is not None and f.sparse_rank is not None
            )

        with self.tracer.span(trace, "rerank", reranker=self.reranker.name) as span:
            result = self.reranker.rerank(question, fused, settings.rerank_top_n)
            span.tokens_in = result.tokens_in
            span.tokens_out = result.tokens_out
            span.metadata["returned"] = len(result.chunks)

        return RetrievalOutcome(
            chunks=result.chunks,
            trace=trace,
            embed_tokens=embedded.tokens,
            rerank_tokens_in=result.tokens_in,
            rerank_tokens_out=result.tokens_out,
        )

    # ----------------------------------------------------------------- answer

    def answer(self, question: str) -> Answer:
        with self.tracer.trace("query", question=question) as trace:
            outcome = self.retrieve(question, trace)
            context = self._context(outcome.chunks)

            with self.tracer.span(trace, "generate", model=self.settings.chat_model) as span:
                reply = self.chat.complete(
                    system=SYSTEM_PROMPT, user=f"{question}\n\nContext:\n{context}"
                )
                span.tokens_in = reply.tokens_in
                span.tokens_out = reply.tokens_out

            return Answer(
                question=question,
                text=reply.text,
                sources=[
                    {
                        "chunk_id": c.id,
                        "doc_id": c.chunk.doc_id,
                        "ordinal": c.chunk.ordinal,
                        "score": round(c.score, 6),
                        "dense_rank": c.dense_rank,
                        "sparse_rank": c.sparse_rank,
                        "text": c.chunk.text,
                    }
                    for c in outcome.chunks
                ],
                trace=trace.as_dict(),
                cost_usd=cost_usd(
                    self.settings,
                    embed_tokens=outcome.embed_tokens,
                    tokens_in=trace.tokens_in - outcome.embed_tokens,
                    tokens_out=trace.tokens_out,
                ),
                latency_ms=trace.duration_ms,
            )

    def _context(self, chunks: list[FusedChunk]) -> str:
        budget = self.settings.max_context_chars
        parts: list[str] = []
        used = 0
        for chunk in chunks:
            body = chunk.chunk.text
            if used + len(body) > budget:
                body = body[: max(0, budget - used)]
            if not body:
                break
            parts.append(f"[{chunk.chunk.doc_id}#{chunk.chunk.ordinal}] {body}")
            used += len(body)
        return "\n\n".join(parts)

    def stats(self) -> dict:
        return {
            "collection": self.settings.qdrant_collection,
            "chunks": self.store.count(),
            "bm25_documents": len(self.bm25),
            "embed_provider": self.settings.resolved_embed_provider,
            "chat_provider": self.settings.resolved_chat_provider,
            "embed_model": self.settings.embed_model,
            "chat_model": self.settings.chat_model,
            "reranker": self.reranker.name,
            "tracing": "langfuse" if self.tracer.enabled else "local-only",
        }
