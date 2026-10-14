"""API tests. The pipeline dependency is swapped for an offline one."""

import pytest
from fastapi.testclient import TestClient

from rag import api
from rag.config import load_settings
from rag.pipeline import RagPipeline


@pytest.fixture
def client(monkeypatch):
    """
    Swap the module-level pipeline factory for an offline one.

    monkeypatch restores the original afterwards, so tests stay independent —
    patching api.get_pipeline by assignment leaks into the next test and
    breaks its cache_clear call.
    """
    settings = load_settings(
        provider="fake", qdrant_url=":memory:", qdrant_collection="api_test"
    )
    pipeline = RagPipeline(settings)
    monkeypatch.setattr(api, "get_pipeline", lambda: pipeline)
    with TestClient(api.app) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ingest_then_query(client):
    ingested = client.post(
        "/ingest",
        json={"doc_id": "alarms", "text": "Fault F-310 overpressure. The default limit is 16 bar."},
    )
    assert ingested.status_code == 200
    assert ingested.json()["chunks_indexed"] >= 1

    response = client.post("/query", json={"question": "What is the overpressure limit?"})
    assert response.status_code == 200
    payload = response.json()
    assert "16" in payload["answer"]
    assert payload["sources"]
    assert [s["name"] for s in payload["trace"]["spans"]][0] == "embed"
    assert payload["latency_ms"] > 0


def test_retrieve_skips_generation(client):
    client.post("/ingest", json={"doc_id": "net", "text": "OPC UA listens on port 4840."})
    payload = client.post("/retrieve", json={"question": "OPC UA port"}).json()
    assert payload["chunks"]
    assert "generate" not in [s["name"] for s in payload["trace"]["spans"]]


def test_ingest_requires_doc_id_with_raw_text(client):
    assert client.post("/ingest", json={"text": "orphan text"}).status_code == 422


def test_ingest_rejects_an_empty_request(client):
    assert client.post("/ingest", json={}).status_code == 422


def test_ingest_reports_a_missing_path(client):
    assert client.post("/ingest", json={"path": "/no/such/place"}).status_code == 404


def test_empty_question_is_rejected(client):
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_stats_exposes_the_active_configuration(client):
    stats = client.get("/stats").json()
    assert stats["provider"] == "fake"
    assert "reranker" in stats


def test_chunk_lookup_returns_neighbours(client):
    client.post(
        "/ingest",
        json={
            "doc_id": "manual",
            "text": "\n\n".join(f"Paragraph number {i} about pumps and valves." for i in range(8)),
        },
    )
    hit = client.post("/retrieve", json={"question": "pumps and valves"}).json()["chunks"][0]

    payload = client.get(f"/chunks/{hit['chunk_id']}", params={"context": 1}).json()
    assert payload["chunk_id"] == hit["chunk_id"]
    ordinals = [c["ordinal"] for c in payload["chunks"]]
    assert ordinals == sorted(ordinals)
    assert payload["ordinal"] in ordinals


def test_chunk_lookup_clamps_context(client):
    client.post("/ingest", json={"doc_id": "d", "text": "Only one short chunk here."})
    hit = client.post("/retrieve", json={"question": "short chunk"}).json()["chunks"][0]
    payload = client.get(f"/chunks/{hit['chunk_id']}", params={"context": 99}).json()
    assert payload["context"] == 5


def test_unknown_chunk_id_is_404_not_a_crash(client):
    assert client.get("/chunks/" + "a" * 32).status_code == 404
    assert client.get("/chunks/not-a-digest").status_code == 404


def test_document_endpoint_returns_chunks_in_order(client):
    client.post(
        "/ingest",
        json={"doc_id": "ordered", "text": "\n\n".join(f"Section {i} text." for i in range(6))},
    )
    payload = client.get("/documents/ordered").json()
    assert payload["doc_id"] == "ordered"
    ordinals = [c["ordinal"] for c in payload["chunks"]]
    assert ordinals == sorted(ordinals)


def test_unknown_document_is_404(client):
    assert client.get("/documents/nope").status_code == 404
