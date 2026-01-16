"""Tests for chunk types: CharacterSpan, Chunk, and PositionAwareChunk."""

import pytest
from pydantic import ValidationError

from rag_evaluation_system.types import (
    CharacterSpan,
    Chunk,
    ChunkId,
    DocumentId,
    PositionAwareChunk,
    PositionAwareChunkId,
)


class TestCharacterSpan:
    """Tests for the CharacterSpan model."""

    def test_character_span_creation_valid(self) -> None:
        """Test CharacterSpan creation with valid data."""
        span = CharacterSpan(
            doc_id=DocumentId("doc1.md"),
            start=0,
            end=5,
            text="Hello",
        )

        assert span.doc_id == DocumentId("doc1.md")
        assert span.start == 0
        assert span.end == 5
        assert span.text == "Hello"

    def test_character_span_end_greater_than_start_validation(self) -> None:
        """Test that end must be greater than start."""
        with pytest.raises(ValueError, match="must be greater than start"):
            CharacterSpan(
                doc_id=DocumentId("doc.md"),
                start=10,
                end=5,  # Invalid: end < start
                text="Hello",
            )

    def test_character_span_end_equals_start_validation(self) -> None:
        """Test that end cannot equal start (empty span not allowed)."""
        with pytest.raises(ValueError, match="must be greater than start"):
            CharacterSpan(
                doc_id=DocumentId("doc.md"),
                start=5,
                end=5,  # Invalid: end == start
                text="",
            )

    def test_character_span_text_length_validation(self) -> None:
        """Test that text length must match span length (end - start)."""
        with pytest.raises(ValueError, match="does not match span length"):
            CharacterSpan(
                doc_id=DocumentId("doc.md"),
                start=0,
                end=10,  # Span length is 10
                text="Short",  # Text length is 5
            )

    def test_character_span_text_length_too_long(self) -> None:
        """Test validation when text is longer than span."""
        with pytest.raises(ValueError, match="does not match span length"):
            CharacterSpan(
                doc_id=DocumentId("doc.md"),
                start=0,
                end=5,  # Span length is 5
                text="Too long text",  # Text is longer
            )

    def test_character_span_length_property(self) -> None:
        """Test the length property returns correct span length."""
        span = CharacterSpan(
            doc_id=DocumentId("doc.md"),
            start=10,
            end=25,
            text="a" * 15,
        )

        assert span.length == 15

    def test_character_span_length_property_large_span(self) -> None:
        """Test length property with large span."""
        text = "x" * 1000
        span = CharacterSpan(
            doc_id=DocumentId("doc.md"),
            start=0,
            end=1000,
            text=text,
        )

        assert span.length == 1000

    def test_character_span_overlaps_same_document_overlapping(self) -> None:
        """Test overlaps returns True for overlapping spans in same document."""
        span1 = CharacterSpan(
            doc_id=DocumentId("doc.md"),
            start=0,
            end=10,
            text="a" * 10,
        )
        span2 = CharacterSpan(
            doc_id=DocumentId("doc.md"),
            start=5,
            end=15,
            text="b" * 10,
        )

        assert span1.overlaps(span2) is True
        assert span2.overlaps(span1) is True

    def test_character_span_overlaps_same_document_not_overlapping(self) -> None:
        """Test overlaps returns False for non-overlapping spans."""
        span1 = CharacterSpan(
            doc_id=DocumentId("doc.md"),
            start=0,
            end=10,
            text="a" * 10,
        )
        span2 = CharacterSpan(
            doc_id=DocumentId("doc.md"),
            start=15,
            end=25,
            text="b" * 10,
        )

        assert span1.overlaps(span2) is False
        assert span2.overlaps(span1) is False

    def test_character_span_overlaps_adjacent_spans(self) -> None:
        """Test overlaps returns False for adjacent (touching) spans."""
        span1 = CharacterSpan(
            doc_id=DocumentId("doc.md"),
            start=0,
            end=10,
            text="a" * 10,
        )
        span2 = CharacterSpan(
            doc_id=DocumentId("doc.md"),
            start=10,  # Starts exactly where span1 ends
            end=20,
            text="b" * 10,
        )

        assert span1.overlaps(span2) is False
        assert span2.overlaps(span1) is False

    def test_character_span_overlaps_different_documents(self) -> None:
        """Test overlaps returns False for spans in different documents."""
        span1 = CharacterSpan(
            doc_id=DocumentId("doc1.md"),
            start=0,
            end=10,
            text="a" * 10,
        )
        span2 = CharacterSpan(
            doc_id=DocumentId("doc2.md"),
            start=5,  # Would overlap if same doc
            end=15,
            text="b" * 10,
        )

        assert span1.overlaps(span2) is False
        assert span2.overlaps(span1) is False

    def test_character_span_overlaps_contained_span(self) -> None:
        """Test overlaps when one span contains the other."""
        outer = CharacterSpan(
            doc_id=DocumentId("doc.md"),
            start=0,
            end=100,
            text="a" * 100,
        )
        inner = CharacterSpan(
            doc_id=DocumentId("doc.md"),
            start=25,
            end=75,
            text="b" * 50,
        )

        assert outer.overlaps(inner) is True
        assert inner.overlaps(outer) is True

    def test_character_span_overlap_chars_overlapping(self) -> None:
        """Test overlap_chars returns correct count for overlapping spans."""
        span1 = CharacterSpan(
            doc_id=DocumentId("doc.md"),
            start=0,
            end=10,
            text="a" * 10,
        )
        span2 = CharacterSpan(
            doc_id=DocumentId("doc.md"),
            start=5,
            end=15,
            text="b" * 10,
        )

        # Overlap is from position 5 to 10 = 5 characters
        assert span1.overlap_chars(span2) == 5
        assert span2.overlap_chars(span1) == 5

    def test_character_span_overlap_chars_no_overlap(self) -> None:
        """Test overlap_chars returns 0 for non-overlapping spans."""
        span1 = CharacterSpan(
            doc_id=DocumentId("doc.md"),
            start=0,
            end=10,
            text="a" * 10,
        )
        span2 = CharacterSpan(
            doc_id=DocumentId("doc.md"),
            start=20,
            end=30,
            text="b" * 10,
        )

        assert span1.overlap_chars(span2) == 0
        assert span2.overlap_chars(span1) == 0

    def test_character_span_overlap_chars_different_documents(self) -> None:
        """Test overlap_chars returns 0 for different documents."""
        span1 = CharacterSpan(
            doc_id=DocumentId("doc1.md"),
            start=0,
            end=10,
            text="a" * 10,
        )
        span2 = CharacterSpan(
            doc_id=DocumentId("doc2.md"),
            start=0,
            end=10,
            text="b" * 10,
        )

        assert span1.overlap_chars(span2) == 0

    def test_character_span_overlap_chars_contained(self) -> None:
        """Test overlap_chars when one span contains another."""
        outer = CharacterSpan(
            doc_id=DocumentId("doc.md"),
            start=0,
            end=100,
            text="a" * 100,
        )
        inner = CharacterSpan(
            doc_id=DocumentId("doc.md"),
            start=25,
            end=75,
            text="b" * 50,
        )

        # Overlap is the entire inner span = 50 characters
        assert outer.overlap_chars(inner) == 50
        assert inner.overlap_chars(outer) == 50

    def test_character_span_merge_overlapping(self) -> None:
        """Test merge creates a span covering both input spans."""
        span1 = CharacterSpan(
            doc_id=DocumentId("doc.md"),
            start=0,
            end=10,
            text="a" * 10,
        )
        span2 = CharacterSpan(
            doc_id=DocumentId("doc.md"),
            start=5,
            end=15,
            text="b" * 10,
        )

        merged = span1.merge(span2)

        assert merged.doc_id == DocumentId("doc.md")
        assert merged.start == 0
        assert merged.end == 15
        assert merged.length == 15
        assert merged.text == "_" * 15  # Placeholder text

    def test_character_span_merge_contained(self) -> None:
        """Test merge when one span contains another."""
        outer = CharacterSpan(
            doc_id=DocumentId("doc.md"),
            start=0,
            end=100,
            text="a" * 100,
        )
        inner = CharacterSpan(
            doc_id=DocumentId("doc.md"),
            start=25,
            end=75,
            text="b" * 50,
        )

        merged = outer.merge(inner)

        # Result should match the outer span dimensions
        assert merged.start == 0
        assert merged.end == 100
        assert merged.length == 100

    def test_character_span_merge_non_overlapping_raises(self) -> None:
        """Test merge raises ValueError for non-overlapping spans."""
        span1 = CharacterSpan(
            doc_id=DocumentId("doc.md"),
            start=0,
            end=10,
            text="a" * 10,
        )
        span2 = CharacterSpan(
            doc_id=DocumentId("doc.md"),
            start=20,
            end=30,
            text="b" * 10,
        )

        with pytest.raises(ValueError, match="non-overlapping"):
            span1.merge(span2)

    def test_character_span_merge_different_documents_raises(self) -> None:
        """Test merge raises ValueError for different documents."""
        span1 = CharacterSpan(
            doc_id=DocumentId("doc1.md"),
            start=0,
            end=10,
            text="a" * 10,
        )
        span2 = CharacterSpan(
            doc_id=DocumentId("doc2.md"),
            start=5,
            end=15,
            text="b" * 10,
        )

        with pytest.raises(ValueError, match="different documents"):
            span1.merge(span2)


class TestChunk:
    """Tests for the Chunk model."""

    def test_chunk_creation_valid(self) -> None:
        """Test Chunk creation with valid data."""
        chunk = Chunk(
            id=ChunkId("chunk_abc123def456"),
            content="This is chunk content.",
            doc_id=DocumentId("doc.md"),
            metadata={"section": "intro"},
        )

        assert chunk.id == ChunkId("chunk_abc123def456")
        assert chunk.content == "This is chunk content."
        assert chunk.doc_id == DocumentId("doc.md")
        assert chunk.metadata == {"section": "intro"}

    def test_chunk_creation_minimal(self) -> None:
        """Test Chunk creation with minimal required fields."""
        chunk = Chunk(
            id=ChunkId("chunk_xyz789"),
            content="Minimal chunk",
            doc_id=DocumentId("doc.md"),
        )

        assert chunk.id == ChunkId("chunk_xyz789")
        assert chunk.content == "Minimal chunk"
        assert chunk.metadata == {}  # Default empty dict

    def test_chunk_immutability(self) -> None:
        """Test that Chunk is frozen (immutable)."""
        chunk = Chunk(
            id=ChunkId("chunk_test"),
            content="Test content",
            doc_id=DocumentId("doc.md"),
        )

        with pytest.raises(ValidationError):  # Pydantic frozen model raises ValidationError
            chunk.content = "New content"  # type: ignore[misc]


class TestPositionAwareChunk:
    """Tests for the PositionAwareChunk model."""

    def test_position_aware_chunk_creation_valid(self) -> None:
        """Test PositionAwareChunk creation with valid data."""
        content = "Hello, World!"
        chunk = PositionAwareChunk(
            id=PositionAwareChunkId("pa_chunk_abc123"),
            content=content,
            doc_id=DocumentId("doc.md"),
            start=0,
            end=len(content),
            metadata={"chunk_index": 0},
        )

        assert chunk.id == PositionAwareChunkId("pa_chunk_abc123")
        assert chunk.content == content
        assert chunk.doc_id == DocumentId("doc.md")
        assert chunk.start == 0
        assert chunk.end == 13
        assert chunk.metadata == {"chunk_index": 0}

    def test_position_aware_chunk_end_greater_than_start_validation(self) -> None:
        """Test that end must be greater than start."""
        with pytest.raises(ValueError, match="must be greater than start"):
            PositionAwareChunk(
                id=PositionAwareChunkId("pa_chunk_test"),
                content="Hello",
                doc_id=DocumentId("doc.md"),
                start=10,
                end=5,  # Invalid: end < start
            )

    def test_position_aware_chunk_content_length_validation(self) -> None:
        """Test that content length must match span length."""
        with pytest.raises(ValueError, match="does not match span length"):
            PositionAwareChunk(
                id=PositionAwareChunkId("pa_chunk_test"),
                content="Short",  # 5 characters
                doc_id=DocumentId("doc.md"),
                start=0,
                end=20,  # Span length is 20
            )

    def test_position_aware_chunk_to_span(self) -> None:
        """Test to_span converts chunk to CharacterSpan correctly."""
        content = "Test content for span conversion."
        chunk = PositionAwareChunk(
            id=PositionAwareChunkId("pa_chunk_xyz"),
            content=content,
            doc_id=DocumentId("doc.md"),
            start=100,
            end=100 + len(content),
        )

        span = chunk.to_span()

        assert isinstance(span, CharacterSpan)
        assert span.doc_id == chunk.doc_id
        assert span.start == chunk.start
        assert span.end == chunk.end
        assert span.text == chunk.content

    def test_position_aware_chunk_to_span_preserves_properties(self) -> None:
        """Test that to_span preserves all relevant properties."""
        content = "x" * 50
        chunk = PositionAwareChunk(
            id=PositionAwareChunkId("pa_chunk_test"),
            content=content,
            doc_id=DocumentId("important_doc.md"),
            start=500,
            end=550,
        )

        span = chunk.to_span()

        assert span.length == chunk.end - chunk.start
        assert span.length == len(content)

    def test_position_aware_chunk_immutability(self) -> None:
        """Test that PositionAwareChunk is frozen (immutable)."""
        chunk = PositionAwareChunk(
            id=PositionAwareChunkId("pa_chunk_frozen"),
            content="Frozen",
            doc_id=DocumentId("doc.md"),
            start=0,
            end=6,
        )

        with pytest.raises(ValidationError):  # Pydantic frozen model raises ValidationError
            chunk.start = 10  # type: ignore[misc]
