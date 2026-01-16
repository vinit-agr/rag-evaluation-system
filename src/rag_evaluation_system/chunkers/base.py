"""Abstract base classes for chunker interfaces.

This module defines two chunker interfaces:
- Chunker: Simple interface returning plain text chunks
- PositionAwareChunker: Full interface returning chunks with position tracking

The adapter pattern (ChunkerPositionAdapter) bridges these two interfaces,
allowing any Chunker to be used for token-level evaluation.
"""

from abc import ABC, abstractmethod

from rag_evaluation_system.types.chunks import PositionAwareChunk
from rag_evaluation_system.types.documents import Document


class Chunker(ABC):
    """Base chunker interface - returns text chunks without position tracking.

    Use this for chunk-level evaluation or when you don't need character
    position information. Simpler to implement than PositionAwareChunker.

    Example:
        >>> class SimpleChunker(Chunker):
        ...     @property
        ...     def name(self) -> str:
        ...         return "simple"
        ...
        ...     def chunk(self, text: str) -> list[str]:
        ...         return text.split("\\n\\n")
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a descriptive name for this chunker.

        The name should be unique and descriptive, suitable for use in
        logging, metrics reporting, and experiment tracking.

        Returns:
            A string identifier for this chunker configuration.
        """
        ...

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Split text into chunks.

        Args:
            text: The full text to chunk.

        Returns:
            List of chunk text strings.
        """
        ...


class PositionAwareChunker(ABC):
    """Chunker that tracks character positions in the source document.

    Required for token-level evaluation where we need to compute
    character-level overlap between retrieved and relevant content.

    Example:
        >>> class MyPositionAwareChunker(PositionAwareChunker):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my-chunker"
        ...
        ...     def chunk_with_positions(self, doc: Document) -> list[PositionAwareChunk]:
        ...         # Implementation that tracks positions
        ...         ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a descriptive name for this chunker.

        The name should be unique and descriptive, suitable for use in
        logging, metrics reporting, and experiment tracking.

        Returns:
            A string identifier for this chunker configuration.
        """
        ...

    @abstractmethod
    def chunk_with_positions(self, doc: Document) -> list[PositionAwareChunk]:
        """Split document into position-aware chunks.

        Args:
            doc: The document to chunk.

        Returns:
            List of PositionAwareChunk objects with character positions.
        """
        ...


__all__ = [
    "Chunker",
    "PositionAwareChunker",
]
