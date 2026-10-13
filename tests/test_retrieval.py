from rag.chunking import Chunk
from rag.retrieval import BM25Index, reciprocal_rank_fusion
from rag.store import ScoredChunk


def make(chunk_id: str, text: str = "text", doc: str = "doc") -> Chunk:
    return Chunk(id=chunk_id, doc_id=doc, ordinal=0, text=text)


def scored(ids: list[str]) -> list[ScoredChunk]:
    return [ScoredChunk(chunk=make(i), score=1.0) for i in ids]


def test_fusion_rewards_agreement_between_lists():
    dense = scored(["a", "b", "c"])
    sparse = scored(["c", "d", "a"])
    fused = reciprocal_rank_fusion(dense, sparse, k=60)
    # "a" is rank 1 and rank 3; "c" is rank 3 and rank 1 — both appear twice
    # and must outrank anything seen by only one retriever.
    top_two = {f.id for f in fused[:2]}
    assert top_two == {"a", "c"}
    assert fused[-1].id in {"b", "d"}


def test_fusion_keeps_documents_found_by_only_one_retriever():
    fused = reciprocal_rank_fusion(scored(["a"]), scored(["b"]), k=60)
    assert {f.id for f in fused} == {"a", "b"}


def test_fusion_records_both_ranks():
    fused = reciprocal_rank_fusion(scored(["a", "b"]), scored(["b"]), k=60)
    by_id = {f.id: f for f in fused}
    assert by_id["b"].dense_rank == 2
    assert by_id["b"].sparse_rank == 1
    assert by_id["a"].sparse_rank is None


def test_fusion_handles_empty_inputs():
    assert reciprocal_rank_fusion([], [], k=60) == []
    assert len(reciprocal_rank_fusion(scored(["a"]), [], k=60)) == 1


def test_fusion_limit_truncates():
    fused = reciprocal_rank_fusion(scored(["a", "b", "c"]), scored([]), k=60, limit=2)
    assert len(fused) == 2


def test_fusion_is_deterministic_on_ties():
    dense, sparse = scored(["a", "b"]), scored(["a", "b"])
    first = [f.id for f in reciprocal_rank_fusion(dense, sparse, k=60)]
    second = [f.id for f in reciprocal_rank_fusion(dense, sparse, k=60)]
    assert first == second


def test_bm25_finds_the_lexically_matching_chunk():
    chunks = [
        make("1", "The dry run fault stops all pumps immediately."),
        make("2", "Firmware updates use a USB stick formatted FAT32."),
        make("3", "Duty rotation selects the pump with the fewest hours."),
    ]
    hits = BM25Index(chunks).search("dry run fault", top_k=2)
    assert hits
    assert hits[0].id == "1"


def test_bm25_on_empty_corpus_returns_nothing():
    assert BM25Index([]).search("anything", top_k=5) == []


def test_bm25_drops_zero_score_hits():
    chunks = [make("1", "completely unrelated vocabulary here")]
    assert BM25Index(chunks).search("quantum chromodynamics", top_k=5) == []
