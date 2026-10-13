import pytest

from rag.chunking import chunk_text


def test_empty_input_produces_no_chunks():
    assert chunk_text("", doc_id="d") == []
    assert chunk_text("   \n\n  \t ", doc_id="d") == []


def test_chunks_respect_the_size_budget():
    text = "\n\n".join(f"Paragraph number {i} with some filler words." * 3 for i in range(40))
    chunks = chunk_text(text, doc_id="d", size=300, overlap=50)
    assert chunks
    assert all(len(c.text) <= 300 + 50 for c in chunks)


def test_oversized_paragraph_is_split_not_dropped():
    paragraph = "x" * 2500
    chunks = chunk_text(paragraph, doc_id="d", size=500, overlap=100)
    assert len(chunks) > 1
    assert sum(len(c.text) for c in chunks) >= 2500


def test_ids_are_content_derived_and_stable():
    text = "First paragraph.\n\nSecond paragraph."
    first = chunk_text(text, doc_id="d", size=100, overlap=10)
    second = chunk_text(text, doc_id="d", size=100, overlap=10)
    assert [c.id for c in first] == [c.id for c in second]


def test_changing_content_changes_ids():
    a = chunk_text("Alpha text here.", doc_id="d", size=100, overlap=10)
    b = chunk_text("Beta text here.", doc_id="d", size=100, overlap=10)
    assert a[0].id != b[0].id


def test_ordinals_are_sequential():
    text = "\n\n".join(f"Paragraph {i}." for i in range(10))
    chunks = chunk_text(text, doc_id="d", size=40, overlap=5)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, 150), (100, -1)])
def test_invalid_parameters_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", doc_id="d", size=size, overlap=overlap)
