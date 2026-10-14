"""
End-to-end tests against an in-process Qdrant with the deterministic embedder.

No container, no API key, no network — but the same code path as production:
real chunking, real vector search, real BM25, real fusion, real reranking.
"""

import pytest

from rag.config import load_settings
from rag.pipeline import RagPipeline
from rag.rerank import IdentityReranker, LexicalReranker
from rag.retrieval import FusedChunk
from rag.chunking import Chunk

CORPUS = {
    "alarms": (
        "Fault F-101 dry run. Suction pressure stayed below 0.4 bar for longer "
        "than the dry run delay. The controller stops all pumps immediately.\n\n"
        "Fault F-310 overpressure. Discharge pressure exceeded 16 bar."
    ),
    "maintenance": (
        "Firmware is updated from a USB stick formatted FAT32. The file must be "
        "named vx7.fw and placed in the root of the stick.\n\n"
        "Mechanical seal inspection is recommended every 8000 hours."
    ),
    "networking": (
        "The OPC UA server listens on port 4840 when the integration licence is "
        "installed. MQTT is publish-only: commands cannot be sent to the "
        "controller over MQTT."
    ),
}


@pytest.fixture
def pipeline():
    settings = load_settings(
        provider="fake", qdrant_url=":memory:", qdrant_collection="test", rerank=True
    )
    pipe = RagPipeline(settings)
    for doc_id, text in CORPUS.items():
        pipe.ingest_text(text, doc_id=doc_id)
    return pipe


def test_ingestion_indexes_every_document(pipeline):
    assert pipeline.store.count() >= len(CORPUS)
    assert len(pipeline.bm25) == pipeline.store.count()


def test_ingestion_is_idempotent(pipeline):
    before = pipeline.store.count()
    for doc_id, text in CORPUS.items():
        pipeline.ingest_text(text, doc_id=doc_id)
    assert pipeline.store.count() == before


def test_retrieval_finds_the_right_document(pipeline):
    outcome = pipeline.retrieve("What triggers the dry run fault?")
    assert outcome.chunks
    assert outcome.chunks[0].chunk.doc_id == "alarms"


def test_lexical_question_reaches_the_right_document(pipeline):
    outcome = pipeline.retrieve("What must the firmware file be named?")
    assert "maintenance" in {c.chunk.doc_id for c in outcome.chunks}


def test_every_stage_is_traced(pipeline):
    outcome = pipeline.retrieve("overpressure limit")
    names = [span.name for span in outcome.trace.spans]
    assert names == ["embed", "search_dense", "search_sparse", "fuse", "rerank"]
    assert all(span.duration_ms >= 0 for span in outcome.trace.spans)


def test_answer_carries_sources_trace_and_cost(pipeline):
    answer = pipeline.answer("On which port does the OPC UA server listen?")
    assert answer.sources
    assert "4840" in answer.text
    assert [s["name"] for s in answer.trace["spans"]][-1] == "generate"
    assert answer.cost_usd >= 0.0
    assert answer.latency_ms > 0.0


def test_results_do_not_exceed_rerank_top_n(pipeline):
    outcome = pipeline.retrieve("pumps")
    assert len(outcome.chunks) <= pipeline.settings.rerank_top_n


def test_context_respects_the_character_budget(pipeline):
    pipeline.settings.max_context_chars = 120
    outcome = pipeline.retrieve("pumps")
    context = pipeline._context(outcome.chunks)
    # The budget caps chunk text; doc headers add a bounded amount per chunk.
    assert len(context) <= 120 + 40 * len(outcome.chunks)


def test_query_on_empty_index_returns_nothing_rather_than_raising():
    settings = load_settings(provider="fake", qdrant_url=":memory:", qdrant_collection="empty")
    empty = RagPipeline(settings)
    outcome = empty.retrieve("anything at all")
    assert outcome.chunks == []
    answer = empty.answer("anything at all")
    assert isinstance(answer.text, str)


def test_identity_reranker_preserves_fusion_order():
    chunks = [
        FusedChunk(chunk=Chunk(id=str(i), doc_id="d", ordinal=i, text=f"text {i}"),
                   score=1.0 / (i + 1), dense_rank=i + 1, sparse_rank=None)
        for i in range(5)
    ]
    kept = IdentityReranker().rerank("query", chunks, top_n=3).chunks
    assert [c.id for c in kept] == ["0", "1", "2"]


def test_lexical_reranker_promotes_term_coverage():
    chunks = [
        FusedChunk(chunk=Chunk(id="a", doc_id="d", ordinal=0, text="unrelated filler text"),
                   score=0.9, dense_rank=1, sparse_rank=None),
        FusedChunk(chunk=Chunk(id="b", doc_id="d", ordinal=1, text="dry run fault stops pumps"),
                   score=0.1, dense_rank=2, sparse_rank=None),
    ]
    kept = LexicalReranker().rerank("dry run fault", chunks, top_n=2).chunks
    assert kept[0].id == "b"


def test_switching_rerank_off_changes_nothing_else(pipeline):
    settings = load_settings(
        provider="fake", qdrant_url=":memory:", qdrant_collection="test2", rerank=False
    )
    plain = RagPipeline(settings)
    for doc_id, text in CORPUS.items():
        plain.ingest_text(text, doc_id=doc_id)
    assert plain.reranker.name == "identity"
    assert plain.retrieve("dry run").chunks


def test_stats_reports_configuration(pipeline):
    stats = pipeline.stats()
    assert stats["chunks"] == pipeline.store.count()
    assert stats["embed_provider"] == "fake"
    assert stats["chat_provider"] == "fake"
    assert stats["tracing"] == "local-only"


def test_providers_can_be_split():
    """
    The Claude API has no embeddings endpoint, so a Claude setup must embed
    elsewhere. The pipeline has to allow that split without any other change.
    """
    settings = load_settings(
        provider="fake", embed_provider="fake", chat_provider="fake",
        qdrant_url=":memory:", qdrant_collection="split",
    )
    assert settings.resolved_embed_provider == "fake"
    assert settings.resolved_chat_provider == "fake"

    hybrid = load_settings(provider="cloud", embed_provider="local")
    assert hybrid.resolved_embed_provider == "local"
    assert hybrid.resolved_chat_provider == "cloud"
    assert hybrid.embed_model == hybrid.ollama_embed_model
    assert hybrid.chat_model == hybrid.cloud_chat_model


def test_temperature_is_omitted_when_unset():
    """Claude 4.7+ rejects temperature below 1, so the field must be omittable."""
    assert load_settings(cloud_temperature=None).cloud_temperature is None
    assert load_settings().cloud_temperature == 0.0
