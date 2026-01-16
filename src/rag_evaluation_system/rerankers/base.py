"""Abstract base class for rerankers."""
from abc import ABC, abstractmethod

from rag_evaluation_system.types import PositionAwareChunk


class Reranker(ABC):
    """Base class for reranking models."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[PositionAwareChunk],
        top_k: int | None = None,
    ) -> list[PositionAwareChunk]:
        """Rerank chunks based on relevance to query."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a descriptive name for this reranker."""
        raise NotImplementedError
