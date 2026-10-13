import pytest

from eval.metrics import (
    dedupe,
    groundedness,
    hit_at_k,
    keyword_coverage,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    refusal,
)


def test_dedupe_keeps_best_rank_per_document():
    assert dedupe(["a", "a", "b", "a", "c"]) == ["a", "b", "c"]


def test_ndcg_never_exceeds_one_even_with_repeated_documents():
    """
    Regression test. Scoring a chunk-level ranking against document-level
    ground truth counted one relevant document at several ranks and produced
    nDCG above 1.0. Metrics must be fed the deduplicated ranking.
    """
    repeated = ["a", "a", "a", "a", "a"]
    assert ndcg_at_k(dedupe(repeated), {"a"}, k=5) == pytest.approx(1.0)
    assert ndcg_at_k(dedupe(repeated), {"a"}, k=5) <= 1.0


def test_ndcg_rewards_earlier_hits():
    early = ndcg_at_k(["a", "x", "y"], {"a"}, k=3)
    late = ndcg_at_k(["x", "y", "a"], {"a"}, k=3)
    assert early > late
    assert early == pytest.approx(1.0)


def test_recall_and_hit_differ_when_several_documents_are_relevant():
    retrieved = ["a", "x", "y"]
    relevant = {"a", "b"}
    assert hit_at_k(retrieved, relevant, 3) == 1.0
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(0.5)


def test_precision_at_k_uses_the_returned_count():
    assert precision_at_k(["a", "x"], {"a"}, k=2) == pytest.approx(0.5)
    assert precision_at_k([], {"a"}, k=5) == 0.0


def test_reciprocal_rank_and_misses():
    assert reciprocal_rank(["x", "a"], {"a"}) == pytest.approx(0.5)
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_empty_relevant_set_scores_zero_not_crash():
    assert recall_at_k(["a"], set(), 5) == 0.0
    assert ndcg_at_k(["a"], set(), 5) == 0.0


def test_keyword_coverage_is_case_insensitive():
    assert keyword_coverage("Fault F-101 at 0.4 bar", ["f-101", "0.4"]) == pytest.approx(1.0)
    assert keyword_coverage("nothing useful", ["f-101", "0.4"]) == 0.0
    assert keyword_coverage("anything", []) == 1.0


def test_groundedness_detects_vocabulary_not_in_context():
    context = "The dry run fault stops all pumps immediately."
    assert groundedness("The dry run fault stops pumps", context) == pytest.approx(1.0)
    assert groundedness("Bitcoin mining profitability analysis", context) == 0.0


def test_refusal_detects_both_languages():
    assert refusal("The context does not contain that information.")
    assert refusal("В контексте нет информации об этом.")
    assert not refusal("The overpressure limit is 16 bar.")


def test_percentile_is_nearest_rank():
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    assert percentile(values, 50) == 3.0
    assert percentile(values, 95) == 100.0
    assert percentile([], 50) == 0.0
