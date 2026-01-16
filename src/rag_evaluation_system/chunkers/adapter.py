"""Adapter to make any Chunker position-aware."""
import logging

from rag_evaluation_system.types import Document, PositionAwareChunk
from rag_evaluation_system.utils.hashing import generate_pa_chunk_id
from .base import Chunker, PositionAwareChunker

logger = logging.getLogger(__name__)


class ChunkerPositionAdapter(PositionAwareChunker):
    """Adapter that wraps a regular Chunker to make it position-aware."""

    def __init__(self, chunker: Chunker):
        self._chunker = chunker
        self._skipped_chunks: int = 0

    @property
    def name(self) -> str:
        return f"PositionAdapter({self._chunker.name})"

    @property
    def skipped_chunks(self) -> int:
        return self._skipped_chunks

    def chunk_with_positions(self, doc: Document) -> list[PositionAwareChunk]:
        chunks = self._chunker.chunk(doc.content)
        result: list[PositionAwareChunk] = []
        current_pos = 0

        for chunk_text in chunks:
            start = doc.content.find(chunk_text, current_pos)
            if start == -1:
                start = doc.content.find(chunk_text)

            if start == -1:
                logger.warning(
                    "Could not locate chunk in source document '%s'. Skipping. Preview: %s...",
                    doc.id,
                    chunk_text[:50],
                )
                self._skipped_chunks += 1
                continue

            end = start + len(chunk_text)
            result.append(
                PositionAwareChunk(
                    id=generate_pa_chunk_id(chunk_text),
                    content=chunk_text,
                    doc_id=doc.id,
                    start=start,
                    end=end,
                )
            )
            current_pos = end

        return result
