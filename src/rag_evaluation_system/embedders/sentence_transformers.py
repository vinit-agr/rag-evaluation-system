"""Sentence Transformers embeddings implementation.

This module provides an Embedder implementation using sentence-transformers.
Requires the `sentence-transformers` optional dependency:
`pip install rag-evaluation-system[sentence-transformers]`
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rag_evaluation_system.embedders.base import Embedder

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder(Embedder):
    """Embedder implementation using sentence-transformers library.

    This embedder uses Hugging Face sentence-transformers models for
    generating dense vector representations locally without API calls.

    Attributes:
        model_name: The sentence-transformers model name.
        _model: The loaded SentenceTransformer model instance.

    Example:
        >>> embedder = SentenceTransformerEmbedder()  # Uses all-MiniLM-L6-v2
        >>> embedding = embedder.embed_query("What is machine learning?")
        >>> len(embedding)
        384

        >>> embedder = SentenceTransformerEmbedder("all-mpnet-base-v2")
        >>> embedding = embedder.embed_query("What is machine learning?")
        >>> len(embedding)
        768
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        """Initialize the sentence-transformers embedder.

        Args:
            model_name: The sentence-transformers model to use. Popular models:
                - "all-MiniLM-L6-v2" (384 dimensions, fast, default)
                - "all-mpnet-base-v2" (768 dimensions, better quality)
                - "multi-qa-mpnet-base-dot-v1" (768 dimensions, for QA)
                See https://www.sbert.net/docs/pretrained_models.html for more.

        Raises:
            ImportError: If the sentence-transformers package is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers package is required for SentenceTransformerEmbedder. "
                "Install it with: pip install rag-evaluation-system[sentence-transformers] "
                "or: pip install sentence-transformers"
            ) from e

        self.model_name = model_name
        self._model: SentenceTransformer = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension.

        Returns:
            The dimensionality of embeddings produced by the configured model.
        """
        return self._model.get_sentence_embedding_dimension()  # type: ignore[return-value]

    @property
    def name(self) -> str:
        """Return a descriptive name for this embedder.

        Returns:
            A string in the format "SentenceTransformer({model_name})".
        """
        return f"SentenceTransformer({self.model_name})"

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        if not texts:
            return []

        # encode() returns numpy array, convert to list of lists
        embeddings: Any = self._model.encode(texts, convert_to_numpy=True)
        result: list[list[float]] = embeddings.tolist()
        return result

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query.

        Args:
            query: The query text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        return self.embed([query])[0]


__all__ = ["SentenceTransformerEmbedder"]
