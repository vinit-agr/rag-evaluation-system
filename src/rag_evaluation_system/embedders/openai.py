"""OpenAI embeddings implementation."""
from typing import TYPE_CHECKING

from .base import Embedder

if TYPE_CHECKING:
    from openai import OpenAI


class OpenAIEmbedder(Embedder):
    """OpenAI text embeddings."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        client: "OpenAI | None" = None,
    ):
        try:
            from openai import OpenAI as OpenAIClient
        except ImportError as e:
            raise ImportError(
                "OpenAI package required. Install with: "
                "pip install rag-evaluation-system[openai]"
            ) from e

        self._model = model
        self._client = client or OpenAIClient()
        self._dimension = self._get_dimension()

    @property
    def name(self) -> str:
        return f"OpenAI({self._model})"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_dimension(self) -> int:
        known_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return known_dims.get(self._model, 1536)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query])[0]
