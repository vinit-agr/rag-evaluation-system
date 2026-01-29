"""Abstract base class for vector store interfaces.

This module defines the VectorStore interface for storing and retrieving
position-aware chunks using vector similarity search.
"""

from abc import ABC, abstractmethod

from rag_evaluation_system.types.chunks import PositionAwareChunk


class VectorStore(ABC):
    """Abstract base class for vector stores.

    Vector stores index position-aware chunks with their embeddings and
    support similarity search to retrieve the most relevant chunks for
    a given query embedding.

    Example:
        >>> class MyVectorStore(VectorStore):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my-store"
        ...
        ...     def add(
        ...         self,
        ...         chunks: list[PositionAwareChunk],
        ...         embeddings: list[list[float]],
        ...     ) -> None:
        ...         # Store chunks and embeddings
        ...         ...
        ...
        ...     def search(
        ...         self,
        ...         query_embedding: list[float],
        ...         k: int = 5,
        ...     ) -> list[PositionAwareChunk]:
        ...         # Find k most similar chunks
        ...         ...
        ...
        ...     def clear(self) -> None:
        ...         # Remove all stored data
        ...         ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a descriptive name for this vector store.

        The name should be unique and descriptive, suitable for use in
        logging, metrics reporting, and experiment tracking.

        Returns:
            A string identifier for this vector store configuration.
        """
        ...

    @abstractmethod
    def add(
        self,
        chunks: list[PositionAwareChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add chunks and their embeddings to the store.

        Args:
            chunks: List of position-aware chunks to store.
            embeddings: List of embedding vectors corresponding to each chunk.
                Must have the same length as chunks.

        Raises:
            ValueError: If chunks and embeddings have different lengths.
        """
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[PositionAwareChunk]:
        """Search for the k most similar chunks.

        Args:
            query_embedding: The query embedding vector.
            k: Number of results to return. Defaults to 5.

        Returns:
            List of the k most similar PositionAwareChunk objects,
            ordered by decreasing similarity.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all chunks and embeddings from the store.

        After calling clear(), the store should be empty and ready
        to accept new data.
        """
        ...


__all__ = ["VectorStore"]
