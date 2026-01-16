"""Abstract base classes for chunkers."""
from abc import ABC, abstractmethod

from rag_evaluation_system.types import Document, PositionAwareChunk


class Chunker(ABC):
    """Base chunker interface - returns text chunks without position tracking."""

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Split text into chunks."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a descriptive name for this chunker configuration."""
        raise NotImplementedError


class PositionAwareChunker(ABC):
    """Chunker that tracks character positions in the source document."""

    @abstractmethod
    def chunk_with_positions(self, doc: Document) -> list[PositionAwareChunk]:
        """Split document into position-aware chunks."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a descriptive name for this chunker configuration."""
        raise NotImplementedError
