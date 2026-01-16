"""Abstract base class for reranker interfaces.

This module defines the Reranker interface for reranking retrieved chunks
based on relevance to a query. Implementations include Cohere Rerank.
"""

from abc import ABC, abstractmethod

from rag_evaluation_system.types import PositionAwareChunk


class Reranker(ABC):
    """Abstract base class for reranking models.

    Rerankers take a list of retrieved chunks and reorder them based on
    their relevance to a given query. This is typically used as a second
    stage after initial retrieval to improve ranking quality.

    Example:
        >>> class MyReranker(Reranker):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my-reranker"
        ...
        ...     def rerank(
        ...         self,
        ...         query: str,
        ...         chunks: list[PositionAwareChunk],
        ...         top_k: int | None = None,
        ...     ) -> list[PositionAwareChunk]:
        ...         # Rerank chunks based on relevance to query
        ...         ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a descriptive name for this reranker.

        The name should be unique and descriptive, suitable for use in
        logging, metrics reporting, and experiment tracking.

        Returns:
            A string identifier for this reranker configuration.
        """
        ...

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[PositionAwareChunk],
        top_k: int | None = None,
    ) -> list[PositionAwareChunk]:
        """Rerank chunks based on relevance to the query.

        Takes a list of chunks and reorders them based on their semantic
        relevance to the given query. Optionally limits the number of
        returned chunks.

        Args:
            query: The query text to rank chunks against.
            chunks: List of chunks to rerank.
            top_k: Optional limit on the number of chunks to return.
                If None, all chunks are returned in reranked order.

        Returns:
            List of chunks reordered by relevance to the query. If top_k
            is specified, only the top_k most relevant chunks are returned.
        """
        ...


__all__ = ["Reranker"]
