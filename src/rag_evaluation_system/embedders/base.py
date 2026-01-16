"""Abstract base class for embedders."""
from abc import ABC, abstractmethod


class Embedder(ABC):
    """Base class for text embedding models."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a descriptive name for this embedder."""
        raise NotImplementedError
