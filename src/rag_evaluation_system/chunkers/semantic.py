"""Semantic chunker placeholder implementation."""
from rag_evaluation_system.types import Document, PositionAwareChunk
from .base import Chunker, PositionAwareChunker
from .recursive_character import RecursiveCharacterChunker


class SemanticChunker(Chunker, PositionAwareChunker):
    """Split text using heuristic semantic boundaries.

    This placeholder implementation delegates to RecursiveCharacterChunker until
    embedding-based segmentation is implemented.
    """

    def __init__(
        self,
        embedder: object | None = None,
        similarity_threshold: float = 0.75,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self._embedder = embedder
        self._similarity_threshold = similarity_threshold
        self._fallback = RecursiveCharacterChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @property
    def name(self) -> str:
        return f"Semantic(threshold={self._similarity_threshold})"

    def chunk(self, text: str) -> list[str]:
        return self._fallback.chunk(text)

    def chunk_with_positions(self, doc: Document) -> list[PositionAwareChunk]:
        return self._fallback.chunk_with_positions(doc)
