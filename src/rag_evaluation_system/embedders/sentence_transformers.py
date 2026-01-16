"""Sentence Transformers embedding implementation."""
from typing import TYPE_CHECKING

from .base import Embedder

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder(Embedder):
    """Local sentence-transformers embedder."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers package required. Install with: "
                "pip install rag-evaluation-system[sentence-transformers]"
            ) from e

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def name(self) -> str:
        return f"SentenceTransformer({self._model_name})"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=False)
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query])[0]
