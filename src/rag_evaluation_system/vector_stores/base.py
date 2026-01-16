"""Abstract base class for vector stores."""
from abc import ABC, abstractmethod

from rag_evaluation_system.types import PositionAwareChunk


class VectorStore(ABC):
    """Base class for vector stores."""

    @abstractmethod
    def add(self, chunks: list[PositionAwareChunk], embeddings: list[list[float]]) -> None:
        """Add chunks with their embeddings."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query_embedding: list[float], k: int = 5) -> list[PositionAwareChunk]:
        """Search for similar chunks."""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Clear all data from the store."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a descriptive name for this vector store."""
        raise NotImplementedError
