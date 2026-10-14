"""
FastAPI surface.

    POST /ingest    index text or a path on the server
    POST /query     retrieve + generate, returns answer, sources and trace
    POST /retrieve  retrieval only — no generation, no generation cost
    GET  /stats     index size and active configuration
    GET  /health    liveness

The trace travels in the /query response rather than only to Langfuse: a
caller debugging a bad answer should be able to see which stage produced it
without opening another tool.

Run:
    uvicorn rag.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import load_settings
from .pipeline import RagPipeline

app = FastAPI(
    title="RAG service",
    description="Hybrid retrieval with reranking, tracing and a measurable eval suite.",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_pipeline() -> RagPipeline:
    return RagPipeline(load_settings())


class IngestRequest(BaseModel):
    text: str | None = None
    doc_id: str | None = None
    path: str | None = None
    pattern: str = "*.md"
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    chunks_indexed: int
    total_chunks: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    trace: dict
    cost_usd: float
    latency_ms: float


class RetrieveResponse(BaseModel):
    question: str
    chunks: list[dict]
    trace: dict
    latency_ms: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/stats")
def stats() -> dict:
    return get_pipeline().stats()


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    pipeline = get_pipeline()
    if request.text is not None:
        if not request.doc_id:
            raise HTTPException(422, "doc_id is required when posting raw text")
        written = pipeline.ingest_text(request.text, request.doc_id, request.metadata)
    elif request.path:
        target = Path(request.path)
        if not target.exists():
            raise HTTPException(404, f"path not found on the server: {request.path}")
        written = pipeline.ingest_path(target, request.pattern)
    else:
        raise HTTPException(422, "provide either text + doc_id, or path")
    return IngestResponse(chunks_indexed=written, total_chunks=pipeline.store.count())


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    answer = get_pipeline().answer(request.question)
    return QueryResponse(
        question=answer.question,
        answer=answer.text,
        sources=answer.sources,
        trace=answer.trace,
        cost_usd=answer.cost_usd,
        latency_ms=answer.latency_ms,
    )


@app.get("/chunks/{chunk_id}")
def get_chunk(chunk_id: str, context: int = 1) -> dict:
    """
    One chunk by id, with `context` neighbouring chunks either side.

    Retrieval returns truncated passages; this is how a caller reads around a
    hit without re-running the query. `context` is clamped to 0-5.
    """
    pipeline = get_pipeline()
    chunk = pipeline.store.get(chunk_id)
    if chunk is None:
        raise HTTPException(404, f"no chunk with id {chunk_id}")

    context = max(0, min(5, context))
    siblings = pipeline.store.by_document(chunk.doc_id)
    low, high = chunk.ordinal - context, chunk.ordinal + context
    window = [c for c in siblings if low <= c.ordinal <= high]
    return {
        "chunk_id": chunk.id,
        "doc_id": chunk.doc_id,
        "ordinal": chunk.ordinal,
        "document_chunks": len(siblings),
        "context": context,
        "chunks": [
            {"chunk_id": c.id, "ordinal": c.ordinal, "text": c.text} for c in window
        ],
    }


@app.get("/documents/{doc_id}")
def get_document(doc_id: str) -> dict:
    """Every chunk of one document, in reading order."""
    chunks = get_pipeline().store.by_document(doc_id)
    if not chunks:
        raise HTTPException(404, f"no document with id {doc_id}")
    return {
        "doc_id": doc_id,
        "chunks": [
            {"chunk_id": c.id, "ordinal": c.ordinal, "text": c.text} for c in chunks
        ],
    }


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: QueryRequest) -> RetrieveResponse:
    outcome = get_pipeline().retrieve(request.question)
    return RetrieveResponse(
        question=request.question,
        chunks=[
            {
                "chunk_id": c.id,
                "doc_id": c.chunk.doc_id,
                "ordinal": c.chunk.ordinal,
                "score": round(c.score, 6),
                "text": c.chunk.text,
            }
            for c in outcome.chunks
        ],
        trace=outcome.trace.as_dict(),
        latency_ms=outcome.trace.duration_ms,
    )
