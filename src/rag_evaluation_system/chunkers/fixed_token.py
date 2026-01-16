"""Fixed-token chunker implementation."""
import re

from rag_evaluation_system.types import Document, PositionAwareChunk
from rag_evaluation_system.utils.hashing import generate_pa_chunk_id
from .base import Chunker, PositionAwareChunker


class FixedTokenChunker(Chunker, PositionAwareChunker):
    """Split text by token count using whitespace tokens."""

    def __init__(self, tokens_per_chunk: int = 200, overlap_tokens: int = 50):
        if overlap_tokens >= tokens_per_chunk:
            raise ValueError("overlap_tokens must be less than tokens_per_chunk")
        self._tokens_per_chunk = tokens_per_chunk
        self._overlap_tokens = overlap_tokens

    @property
    def name(self) -> str:
        return f"FixedToken(tokens={self._tokens_per_chunk}, overlap={self._overlap_tokens})"

    def chunk(self, text: str) -> list[str]:
        spans = self._token_spans(text)
        if not spans:
            return []
        chunks: list[str] = []
        for start_idx, end_idx in self._window_indices(len(spans)):
            start = spans[start_idx][0]
            end = spans[end_idx][1]
            chunks.append(text[start:end])
        return chunks

    def chunk_with_positions(self, doc: Document) -> list[PositionAwareChunk]:
        spans = self._token_spans(doc.content)
        if not spans:
            return []
        chunks: list[PositionAwareChunk] = []
        for start_idx, end_idx in self._window_indices(len(spans)):
            start = spans[start_idx][0]
            end = spans[end_idx][1]
            chunk_text = doc.content[start:end]
            chunks.append(
                PositionAwareChunk(
                    id=generate_pa_chunk_id(chunk_text),
                    content=chunk_text,
                    doc_id=doc.id,
                    start=start,
                    end=end,
                )
            )
        return chunks

    def _token_spans(self, text: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    def _window_indices(self, token_count: int) -> list[tuple[int, int]]:
        indices: list[tuple[int, int]] = []
        step = self._tokens_per_chunk - self._overlap_tokens
        for start in range(0, token_count, step):
            end = min(start + self._tokens_per_chunk, token_count)
            if start >= end:
                break
            indices.append((start, end - 1))
            if end == token_count:
                break
        return indices
