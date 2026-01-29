"""OpenAI embeddings implementation.

This module provides an Embedder implementation using OpenAI's embedding API.
Requires the `openai` optional dependency: `pip install rag-evaluation-system[openai]`
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_evaluation_system.embedders.base import Embedder

if TYPE_CHECKING:
    from openai import OpenAI

# Known embedding model dimensions
_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(Embedder):
    """Embedder implementation using OpenAI's embedding API.

    This embedder uses OpenAI's text embedding models to generate dense
    vector representations of text for semantic similarity search.

    Attributes:
        model: The OpenAI embedding model name.
        _client: The OpenAI client instance.
        _dimension: The embedding dimension for the selected model.

    Example:
        >>> embedder = OpenAIEmbedder()  # Uses text-embedding-3-small by default
        >>> embedding = embedder.embed_query("What is machine learning?")
        >>> len(embedding)
        1536

        >>> embedder = OpenAIEmbedder(model="text-embedding-3-large")
        >>> embedding = embedder.embed_query("What is machine learning?")
        >>> len(embedding)
        3072
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        client: OpenAI | None = None,
    ) -> None:
        """Initialize the OpenAI embedder.

        Args:
            model: The OpenAI embedding model to use. Supported models:
                - "text-embedding-3-small" (1536 dimensions, default)
                - "text-embedding-3-large" (3072 dimensions)
                - "text-embedding-ada-002" (1536 dimensions, legacy)
            client: Optional pre-configured OpenAI client. If not provided,
                a new client will be created using environment variables
                (OPENAI_API_KEY).

        Raises:
            ImportError: If the openai package is not installed.
            ValueError: If the model is not recognized (unknown dimension).
        """
        try:
            from openai import OpenAI as OpenAIClient
        except ImportError as e:
            raise ImportError(
                "OpenAI package is required for OpenAIEmbedder. "
                "Install it with: pip install rag-evaluation-system[openai] "
                "or: pip install openai"
            ) from e

        self.model = model
        self._client = client if client is not None else OpenAIClient()

        if model in _MODEL_DIMENSIONS:
            self._dimension = _MODEL_DIMENSIONS[model]
        else:
            raise ValueError(
                f"Unknown OpenAI embedding model: {model}. "
                f"Known models: {list(_MODEL_DIMENSIONS.keys())}. "
                "If this is a new model, please update the _MODEL_DIMENSIONS mapping."
            )

    @property
    def dimension(self) -> int:
        """Return the embedding dimension.

        Returns:
            The dimensionality of embeddings produced by the configured model.
        """
        return self._dimension

    @property
    def name(self) -> str:
        """Return a descriptive name for this embedder.

        Returns:
            A string in the format "OpenAI({model_name})".
        """
        return f"OpenAI({self.model})"

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        if not texts:
            return []

        response = self._client.embeddings.create(
            model=self.model,
            input=texts,
        )

        # Sort by index to ensure correct ordering
        sorted_embeddings = sorted(response.data, key=lambda x: x.index)
        return [embedding.embedding for embedding in sorted_embeddings]

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query.

        Args:
            query: The query text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        return self.embed([query])[0]


__all__ = ["OpenAIEmbedder"]
