"""Cohere reranker implementation.

This module provides a Reranker implementation using Cohere's Rerank API.
Requires the `cohere` optional dependency: `pip install rag-evaluation-system[cohere]`
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_evaluation_system.rerankers.base import Reranker

if TYPE_CHECKING:
    import cohere

    from rag_evaluation_system.types import PositionAwareChunk


class CohereReranker(Reranker):
    """Reranker implementation using Cohere's Rerank API.

    This reranker uses Cohere's neural reranking models to reorder chunks
    based on their semantic relevance to a given query.

    Attributes:
        model: The Cohere rerank model name.
        _client: The Cohere client instance.

    Example:
        >>> reranker = CohereReranker()  # Uses rerank-english-v3.0 by default
        >>> reranked = reranker.rerank("What is machine learning?", chunks)
        >>> # reranked contains chunks ordered by relevance

        >>> reranker = CohereReranker(model="rerank-multilingual-v3.0")
        >>> reranked = reranker.rerank("What is machine learning?", chunks, top_k=5)
        >>> len(reranked)
        5
    """

    def __init__(
        self,
        model: str = "rerank-english-v3.0",
        client: cohere.Client | None = None,
    ) -> None:
        """Initialize the Cohere reranker.

        Args:
            model: The Cohere rerank model to use. Supported models:
                - "rerank-english-v3.0" (default)
                - "rerank-multilingual-v3.0"
                - "rerank-english-v2.0"
                - "rerank-multilingual-v2.0"
            client: Optional pre-configured Cohere client. If not provided,
                a new client will be created using environment variables
                (COHERE_API_KEY).

        Raises:
            ImportError: If the cohere package is not installed.
        """
        try:
            import cohere as cohere_module
        except ImportError as e:
            raise ImportError(
                "Cohere package is required for CohereReranker. "
                "Install it with: pip install rag-evaluation-system[cohere] "
                "or: pip install cohere"
            ) from e

        self.model = model
        self._client = client if client is not None else cohere_module.Client()

    @property
    def name(self) -> str:
        """Return a descriptive name for this reranker.

        Returns:
            A string in the format "Cohere({model_name})".
        """
        return f"Cohere({self.model})"

    def rerank(
        self,
        query: str,
        chunks: list[PositionAwareChunk],
        top_k: int | None = None,
    ) -> list[PositionAwareChunk]:
        """Rerank chunks based on relevance to the query using Cohere.

        Args:
            query: The query text to rank chunks against.
            chunks: List of chunks to rerank.
            top_k: Optional limit on the number of chunks to return.
                If None, all chunks are returned in reranked order.

        Returns:
            List of chunks reordered by relevance to the query.
        """
        if not chunks:
            return []

        # Extract document contents from chunks
        documents = [chunk.content for chunk in chunks]

        # Call Cohere rerank API
        response = self._client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=top_k if top_k is not None else len(chunks),
        )

        # Reorder chunks based on result indices
        reranked_chunks = [chunks[result.index] for result in response.results]

        return reranked_chunks


__all__ = ["CohereReranker"]
