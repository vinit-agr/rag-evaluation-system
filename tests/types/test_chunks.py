"""Tests for chunk and span types."""
import pytest

from rag_evaluation_system.types import CharacterSpan, DocumentId, PositionAwareChunk, PositionAwareChunkId


def test_character_span_length():
    span = CharacterSpan(doc_id=DocumentId("doc1"), start=10, end=50, text="x" * 40)
    assert span.length == 40


def test_character_span_overlap():
    span1 = CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=50, text="x" * 50)
    span2 = CharacterSpan(doc_id=DocumentId("doc1"), start=30, end=80, text="x" * 50)
    assert span1.overlaps(span2)
    assert span1.overlap_chars(span2) == 20


def test_character_span_no_overlap_different_docs():
    span1 = CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=50, text="x" * 50)
    span2 = CharacterSpan(doc_id=DocumentId("doc2"), start=0, end=50, text="x" * 50)
    assert not span1.overlaps(span2)
    assert span1.overlap_chars(span2) == 0


def test_character_span_validation_end_before_start():
    with pytest.raises(ValueError, match="end .* must be greater than start"):
        CharacterSpan(doc_id=DocumentId("doc1"), start=50, end=10, text="x")


def test_character_span_validation_text_length_mismatch():
    with pytest.raises(ValueError, match="text length .* doesn't match"):
        CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=50, text="x" * 10)


def test_position_aware_chunk_validation():
    with pytest.raises(ValueError, match="content length .* doesn't match"):
        PositionAwareChunk(
            id=PositionAwareChunkId("pa_chunk_test"),
            content="short",
            doc_id=DocumentId("doc1"),
            start=0,
            end=10,
        )
