"""Abstract base class for embedder interfaces.

This module defines the Embedder interface for generating vector embeddings
from text. Implementations include OpenAI embeddings and sentence-transformers.
"""

from abc import ABC, abstractmethod


class Embedder(ABC):
    """Abstract base class for text embedding models.

    Embedders transform text into dense vector representations that can be
    used for semantic similarity search in vector stores.

    Example:
        >>> class MyEmbedder(Embedder):
        ...     @property
        ...     def dimension(self) -> int:
        ...         return 768
        ...
        ...     @property
        ...     def name(self) -> str:
        ...         return "my-embedder"
        ...
        ...     def embed(self, texts: list[str]) -> list[list[float]]:
        ...         # Generate embeddings for multiple texts
        ...         ...
        ...
        ...     def embed_query(self, query: str) -> list[float]:
        ...         return self.embed([query])[0]
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension.

        Returns:
            The dimensionality of the embedding vectors produced by this model.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a descriptive name for this embedder.

        The name should be unique and descriptive, suitable for use in
        logging, metrics reporting, and experiment tracking.

        Returns:
            A string identifier for this embedder configuration.
        """
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text. Each embedding
            is a list of floats with length equal to self.dimension.
        """
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query.

        This method is provided for convenience and clarity when embedding
        search queries. Implementations may optimize for single-text embedding.

        Args:
            query: The query text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        ...


__all__ = ["Embedder"]
