"""Chunk and character span models for the RAG evaluation system.

This module defines data structures for representing text chunks and
character-level spans used in both chunk-level and token-level evaluation.
"""

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_evaluation_system.types.primitives import (
    ChunkId,
    DocumentId,
    PositionAwareChunkId,
)


class CharacterSpan(BaseModel):
    """A span of characters in a source document.

    Represents a contiguous range of text within a document, defined by
    start and end character positions. Used in token-level evaluation for:
    - Ground truth data (relevant excerpts from documents)
    - Computing overlap between retrieved chunks and ground truth

    Attributes:
        doc_id: The document this span belongs to.
        start: Starting character position (inclusive, 0-indexed).
        end: Ending character position (exclusive).
        text: The actual text content of this span. Included for convenience
            and validation - should match document[start:end].

    Example:
        For document content "Hello, World!", CharacterSpan("doc1", 0, 5, "Hello")
        represents the text "Hello".
    """

    model_config = ConfigDict(frozen=True)

    doc_id: DocumentId
    start: int = Field(ge=0, description="Starting character position (inclusive, 0-indexed)")
    end: int = Field(ge=0, description="Ending character position (exclusive)")
    text: str

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        """Validate that end > start and text length matches span length."""
        if self.end <= self.start:
            raise ValueError(
                f"End position ({self.end}) must be greater than start position ({self.start})"
            )
        expected_length = self.end - self.start
        actual_length = len(self.text)
        if actual_length != expected_length:
            raise ValueError(
                f"Text length ({actual_length}) does not match span length ({expected_length})"
            )
        return self

    @property
    def length(self) -> int:
        """Return the length of this span in characters."""
        return self.end - self.start

    def overlaps(self, other: "CharacterSpan") -> bool:
        """Check if this span overlaps with another span.

        Two spans overlap if they share at least one character position
        AND belong to the same document.

        Args:
            other: The other span to check for overlap.

        Returns:
            True if spans overlap, False otherwise.
        """
        if self.doc_id != other.doc_id:
            return False
        return self.start < other.end and other.start < self.end

    def overlap_chars(self, other: "CharacterSpan") -> int:
        """Calculate the number of overlapping characters with another span.

        Args:
            other: The other span to calculate overlap with.

        Returns:
            Number of characters in the intersection. Returns 0 if no overlap.
        """
        if not self.overlaps(other):
            return 0
        return min(self.end, other.end) - max(self.start, other.start)

    def merge(self, other: "CharacterSpan") -> "CharacterSpan":
        """Merge this span with another overlapping span.

        Creates a new span that covers the union of both spans. The text
        field will be an empty placeholder since we cannot reconstruct
        the merged text without access to the source document.

        Args:
            other: The other span to merge with. Must overlap with this span.

        Returns:
            A new CharacterSpan covering the union of both spans.

        Raises:
            ValueError: If spans do not overlap or belong to different documents.
        """
        if self.doc_id != other.doc_id:
            raise ValueError(
                f"Cannot merge spans from different documents: {self.doc_id} vs {other.doc_id}"
            )
        if not self.overlaps(other):
            raise ValueError("Cannot merge non-overlapping spans")

        new_start = min(self.start, other.start)
        new_end = max(self.end, other.end)
        new_length = new_end - new_start

        # Create placeholder text of correct length
        # We use a placeholder because we don't have access to the source document
        placeholder_text = "_" * new_length

        return CharacterSpan(
            doc_id=self.doc_id,
            start=new_start,
            end=new_end,
            text=placeholder_text,
        )


class Chunk(BaseModel):
    """A chunk of text extracted from a document (without position tracking).

    Used in chunk-level evaluation where we only care about chunk identity,
    not the exact character positions in the source document.

    Attributes:
        id: Unique identifier for this chunk. Format: "chunk_" + content hash.
            Example: "chunk_a3f2b1c8d9e0"
        content: The actual text content of this chunk.
        doc_id: Reference to the parent document this chunk was extracted from.
        metadata: Arbitrary key-value pairs for additional chunk information.
            Examples: {"chunk_index": 5, "section": "introduction"}
    """

    model_config = ConfigDict(frozen=True)

    id: ChunkId
    content: str
    doc_id: DocumentId
    metadata: dict[str, Any] = Field(default_factory=dict)


class PositionAwareChunk(BaseModel):
    """A chunk that knows its exact position in the source document.

    Used in token-level evaluation at EVALUATION TIME (not data generation).
    When evaluating, chunks are created with position tracking so we can
    compute character-level overlap with ground truth spans.

    Attributes:
        id: Unique identifier for this chunk. Format: "pa_chunk_" + content hash.
            Example: "pa_chunk_7d9e4f2a1b3c"
        content: The actual text content of this chunk.
        doc_id: Reference to the parent document this chunk was extracted from.
        start: Starting character position in the source document (inclusive).
        end: Ending character position in the source document (exclusive).
        metadata: Arbitrary key-value pairs for additional chunk information.

    Note:
        The content should exactly match document[start:end]. This invariant
        is important for correct metric calculation.
    """

    model_config = ConfigDict(frozen=True)

    id: PositionAwareChunkId
    content: str
    doc_id: DocumentId
    start: int = Field(ge=0, description="Starting character position (inclusive, 0-indexed)")
    end: int = Field(ge=0, description="Ending character position (exclusive)")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_positions(self) -> Self:
        """Validate that end > start and content length matches span length."""
        if self.end <= self.start:
            raise ValueError(
                f"End position ({self.end}) must be greater than start position ({self.start})"
            )
        expected_length = self.end - self.start
        actual_length = len(self.content)
        if actual_length != expected_length:
            raise ValueError(
                f"Content length ({actual_length}) does not match span length ({expected_length})"
            )
        return self

    def to_span(self) -> CharacterSpan:
        """Convert this chunk to a CharacterSpan for metric calculation.

        Returns:
            A CharacterSpan with the same document, position, and text info.
        """
        return CharacterSpan(
            doc_id=self.doc_id,
            start=self.start,
            end=self.end,
            text=self.content,
        )


__all__ = [
    "CharacterSpan",
    "Chunk",
    "PositionAwareChunk",
]
