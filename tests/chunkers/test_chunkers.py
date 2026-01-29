"""Tests for chunker interfaces and implementations."""

import pytest

from rag_evaluation_system.chunkers import (
    Chunker,
    ChunkerPositionAdapter,
    PositionAwareChunker,
    RecursiveCharacterChunker,
)
from rag_evaluation_system.types import Document, DocumentId


class TestRecursiveCharacterChunker:
    """Tests for RecursiveCharacterChunker."""

    def test_init_default_values(self) -> None:
        """Test that default values are set correctly."""
        chunker = RecursiveCharacterChunker()
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200
        assert chunker.separators == ["\n\n", "\n", ". ", " ", ""]

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        chunker = RecursiveCharacterChunker(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n", " "],
        )
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50
        assert chunker.separators == ["\n", " "]

    def test_init_invalid_overlap(self) -> None:
        """Test that overlap >= chunk_size raises ValueError."""
        with pytest.raises(ValueError, match=r"chunk_overlap.*must be less than.*chunk_size"):
            RecursiveCharacterChunker(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError, match=r"chunk_overlap.*must be less than.*chunk_size"):
            RecursiveCharacterChunker(chunk_size=100, chunk_overlap=150)

    def test_name_property(self) -> None:
        """Test that name includes size and overlap."""
        chunker = RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50)
        assert chunker.name == "recursive-character(size=500, overlap=50)"

    def test_implements_both_interfaces(self) -> None:
        """Test that RecursiveCharacterChunker implements both interfaces."""
        chunker = RecursiveCharacterChunker()
        assert isinstance(chunker, Chunker)
        assert isinstance(chunker, PositionAwareChunker)

    def test_chunk_empty_text(self) -> None:
        """Test chunking empty text returns empty list."""
        chunker = RecursiveCharacterChunker()
        assert chunker.chunk("") == []

    def test_chunk_small_text(self) -> None:
        """Test chunking text smaller than chunk_size."""
        chunker = RecursiveCharacterChunker(chunk_size=1000, chunk_overlap=100)
        text = "This is a small piece of text."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_paragraph_splitting(self) -> None:
        """Test that paragraphs are split on double newlines."""
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=0)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunker.chunk(text)
        # All paragraphs fit in single chunks due to small size
        assert len(chunks) >= 1
        # Verify text is preserved
        combined = "".join(chunks)
        assert combined == text

    def test_chunk_with_positions_empty_document(self) -> None:
        """Test chunking empty document returns empty list."""
        chunker = RecursiveCharacterChunker()
        doc = Document(id=DocumentId("test"), content="")
        assert chunker.chunk_with_positions(doc) == []

    def test_chunk_with_positions_tracks_positions(self) -> None:
        """Test that position-aware chunking tracks correct positions."""
        chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=0)
        content = "First sentence. Second sentence. Third sentence."
        doc = Document(id=DocumentId("test"), content=content)

        chunks = chunker.chunk_with_positions(doc)

        # Verify positions are correct
        for chunk in chunks:
            assert chunk.doc_id == doc.id
            assert chunk.content == content[chunk.start : chunk.end]
            assert chunk.start >= 0
            assert chunk.end <= len(content)
            assert chunk.end > chunk.start

    def test_chunk_with_positions_all_content_covered(self) -> None:
        """Test that all content is represented in chunks (with possible overlap)."""
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
        content = "A" * 50 + "\n\n" + "B" * 50 + "\n\n" + "C" * 50
        doc = Document(id=DocumentId("test"), content=content)

        chunks = chunker.chunk_with_positions(doc)

        # Combine all chunks should cover all content
        # (Note: with overlap, some parts may be duplicated)
        covered = set()
        for chunk in chunks:
            for i in range(chunk.start, chunk.end):
                covered.add(i)

        assert covered == set(range(len(content)))

    def test_chunk_ids_are_content_based(self) -> None:
        """Test that chunk IDs are deterministic based on content."""
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=0)
        content = "Some test content for chunking."
        doc = Document(id=DocumentId("test"), content=content)

        chunks1 = chunker.chunk_with_positions(doc)
        chunks2 = chunker.chunk_with_positions(doc)

        # Same content should produce same IDs
        assert [c.id for c in chunks1] == [c.id for c in chunks2]

    def test_chunk_ids_have_correct_prefix(self) -> None:
        """Test that position-aware chunk IDs have pa_chunk_ prefix."""
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=0)
        content = "Test content."
        doc = Document(id=DocumentId("test"), content=content)

        chunks = chunker.chunk_with_positions(doc)

        for chunk in chunks:
            assert chunk.id.startswith("pa_chunk_")


class TestChunkerPositionAdapter:
    """Tests for ChunkerPositionAdapter."""

    def test_wraps_simple_chunker(self) -> None:
        """Test that adapter wraps a simple chunker."""
        base_chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=0)
        adapter = ChunkerPositionAdapter(base_chunker)

        assert isinstance(adapter, PositionAwareChunker)

    def test_name_includes_wrapped_chunker(self) -> None:
        """Test that adapter name includes wrapped chunker name."""
        base_chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=0)
        adapter = ChunkerPositionAdapter(base_chunker)

        assert "adapted" in adapter.name
        assert base_chunker.name in adapter.name

    def test_finds_chunk_positions(self) -> None:
        """Test that adapter correctly finds chunk positions."""
        # Create a simple chunker that splits on newlines
        base_chunker = RecursiveCharacterChunker(
            chunk_size=1000, chunk_overlap=0, separators=["\n"]
        )
        adapter = ChunkerPositionAdapter(base_chunker)

        content = "Line one\nLine two\nLine three"
        doc = Document(id=DocumentId("test"), content=content)

        chunks = adapter.chunk_with_positions(doc)

        # Verify positions match content
        for chunk in chunks:
            assert chunk.content == content[chunk.start : chunk.end]

    def test_skipped_chunks_count(self) -> None:
        """Test that skipped chunks are counted."""
        base_chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=0)
        adapter = ChunkerPositionAdapter(base_chunker)

        # Initial count should be 0
        assert adapter.skipped_chunks == 0


class TestHashingFunctions:
    """Tests for hashing utility functions."""

    def test_generate_chunk_id(self) -> None:
        """Test generate_chunk_id produces correct format."""
        from rag_evaluation_system.utils import generate_chunk_id

        chunk_id = generate_chunk_id("test content")
        assert chunk_id.startswith("chunk_")
        # "chunk_" + 12 hex chars = 18 chars
        assert len(chunk_id) == 18

    def test_generate_pa_chunk_id(self) -> None:
        """Test generate_pa_chunk_id produces correct format."""
        from rag_evaluation_system.utils import generate_pa_chunk_id

        chunk_id = generate_pa_chunk_id("test content")
        assert chunk_id.startswith("pa_chunk_")
        # "pa_chunk_" + 12 hex chars = 21 chars
        assert len(chunk_id) == 21

    def test_deterministic_ids(self) -> None:
        """Test that same content produces same IDs."""
        from rag_evaluation_system.utils import generate_chunk_id, generate_pa_chunk_id

        content = "Hello, World!"

        assert generate_chunk_id(content) == generate_chunk_id(content)
        assert generate_pa_chunk_id(content) == generate_pa_chunk_id(content)

    def test_different_content_different_ids(self) -> None:
        """Test that different content produces different IDs."""
        from rag_evaluation_system.utils import generate_chunk_id, generate_pa_chunk_id

        assert generate_chunk_id("content1") != generate_chunk_id("content2")
        assert generate_pa_chunk_id("content1") != generate_pa_chunk_id("content2")
