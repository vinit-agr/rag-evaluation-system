"""Cohere reranker implementation."""
from typing import TYPE_CHECKING

from rag_evaluation_system.types import PositionAwareChunk
from .base import Reranker

if TYPE_CHECKING:
    import cohere


class CohereReranker(Reranker):
    """Cohere reranking model."""

    def __init__(
        self,
        model: str = "rerank-english-v3.0",
        client: "cohere.Client | None" = None,
    ):
        try:
            import cohere
        except ImportError as e:
            raise ImportError(
                "Cohere package required. Install with: "
                "pip install rag-evaluation-system[cohere]"
            ) from e

        self._model = model
        self._client = client or cohere.Client()

    @property
    def name(self) -> str:
        return f"Cohere({self._model})"

    def rerank(
        self,
        query: str,
        chunks: list[PositionAwareChunk],
        top_k: int | None = None,
    ) -> list[PositionAwareChunk]:
        if not chunks:
            return []

        documents = [chunk.content for chunk in chunks]

        response = self._client.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=top_k or len(chunks),
        )

        reranked: list[PositionAwareChunk] = []
        for result in response.results:
            reranked.append(chunks[result.index])

        return reranked
