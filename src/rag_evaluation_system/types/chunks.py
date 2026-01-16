"""Chunk and span types for retrieval evaluation."""
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .primitives import ChunkId, DocumentId, PositionAwareChunkId


class CharacterSpan(BaseModel):
    """A span of characters in a source document."""

    model_config = ConfigDict(frozen=True)

    doc_id: DocumentId
    start: int = Field(ge=0, description="Start position (inclusive, 0-indexed)")
    end: int = Field(ge=0, description="End position (exclusive)")
    text: str = Field(description="The actual text content")

    @model_validator(mode="after")
    def validate_positions(self) -> "CharacterSpan":
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be greater than start ({self.start})")
        expected_length = self.end - self.start
        if len(self.text) != expected_length:
            raise ValueError(
                f"text length ({len(self.text)}) doesn't match span length ({expected_length})"
            )
        return self

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: "CharacterSpan") -> bool:
        if self.doc_id != other.doc_id:
            return False
        return self.start < other.end and other.start < self.end

    def overlap_chars(self, other: "CharacterSpan") -> int:
        if not self.overlaps(other):
            return 0
        return min(self.end, other.end) - max(self.start, other.start)

    def merge(self, other: "CharacterSpan") -> "CharacterSpan":
        if not self.overlaps(other):
            raise ValueError("Cannot merge non-overlapping spans")
        return CharacterSpan(
            doc_id=self.doc_id,
            start=min(self.start, other.start),
            end=max(self.end, other.end),
            text="",
        )


class Chunk(BaseModel):
    """A text chunk without position tracking."""

    model_config = ConfigDict(frozen=True)

    id: ChunkId
    content: str
    doc_id: DocumentId
    metadata: dict[str, Any] = Field(default_factory=dict)


class PositionAwareChunk(BaseModel):
    """A chunk that knows its exact position in the source document."""

    model_config = ConfigDict(frozen=True)

    id: PositionAwareChunkId
    content: str
    doc_id: DocumentId
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_positions(self) -> "PositionAwareChunk":
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be greater than start ({self.start})")
        expected_length = self.end - self.start
        if len(self.content) != expected_length:
            raise ValueError(
                f"content length ({len(self.content)}) doesn't match span ({expected_length})"
            )
        return self

    def to_span(self) -> CharacterSpan:
        return CharacterSpan(
            doc_id=self.doc_id,
            start=self.start,
            end=self.end,
            text=self.content,
        )
