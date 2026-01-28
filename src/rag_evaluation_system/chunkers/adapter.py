"""Adapter to wrap a simple Chunker as a PositionAwareChunker.

This module provides the ChunkerPositionAdapter class that enables using
any existing Chunker implementation for token-level evaluation without
modifying the chunker itself.
"""

import logging

from rag_evaluation_system.chunkers.base import Chunker, PositionAwareChunker
from rag_evaluation_system.types.chunks import PositionAwareChunk
from rag_evaluation_system.types.documents import Document
from rag_evaluation_system.utils.hashing import generate_pa_chunk_id

logger = logging.getLogger(__name__)


class ChunkerPositionAdapter(PositionAwareChunker):
    """Adapter that wraps a regular Chunker to make it position-aware.

    This allows using any existing Chunker implementation for token-level
    evaluation without modifying the chunker itself.

    The adapter works by finding each chunk's text in the source document.
    It searches from the current position first (for efficiency with
    sequential chunks), then falls back to searching from the beginning
    if not found.

    Limitations:
        - May fail if the chunker normalizes whitespace or modifies text
        - May fail if the chunker reorders or combines content
        - Logs a warning and skips chunks that cannot be located

    For best results, use chunkers that preserve the original text exactly.

    Attributes:
        chunker: The wrapped Chunker instance.
        skipped_chunks: Count of chunks that could not be located.

    Example:
        >>> from some_library import ExternalChunker
        >>> chunker = ChunkerPositionAdapter(ExternalChunker())
        >>> chunks = chunker.chunk_with_positions(document)
        >>> if chunker.skipped_chunks > 0:
        ...     print(f"Warning: {chunker.skipped_chunks} chunks were skipped")
    """

    def __init__(self, chunker: Chunker) -> None:
        """Initialize the adapter with a Chunker to wrap.

        Args:
            chunker: The Chunker instance to wrap.
        """
        self._chunker = chunker
        self._skipped_chunks: int = 0

    @property
    def name(self) -> str:
        """Return a descriptive name including the wrapped chunker's name.

        Returns:
            A string identifier in the format "adapted({wrapped_chunker_name})".
        """
        return f"adapted({self._chunker.name})"

    @property
    def skipped_chunks(self) -> int:
        """Return the count of chunks that could not be located.

        This count is cumulative across all calls to chunk_with_positions.
        Reset by creating a new adapter instance.

        Returns:
            The number of chunks that were skipped due to position lookup failure.
        """
        return self._skipped_chunks

    def chunk_with_positions(self, doc: Document) -> list[PositionAwareChunk]:
        """Split document into position-aware chunks using the wrapped chunker.

        For each chunk returned by the wrapped chunker, this method finds
        its position in the source document. Chunks that cannot be located
        are skipped and logged as warnings.

        Args:
            doc: The document to chunk.

        Returns:
            List of PositionAwareChunk objects with character positions.
            Chunks that could not be located are omitted from the result.
        """
        chunks = self._chunker.chunk(doc.content)
        result: list[PositionAwareChunk] = []
        current_pos = 0

        for chunk_text in chunks:
            # Try to find chunk starting from current position (efficient for sequential chunks)
            start = doc.content.find(chunk_text, current_pos)

            if start == -1:
                # Fallback: search from beginning (handles non-sequential chunks)
                start = doc.content.find(chunk_text)

            if start == -1:
                # Chunk text not found - chunker may have modified it
                self._skipped_chunks += 1
                preview = chunk_text[:50] + "..." if len(chunk_text) > 50 else chunk_text
                logger.warning(
                    "Could not locate chunk in source document. "
                    "Chunk may have been modified by chunker. Skipping. "
                    f"Chunk preview: {preview!r}"
                )
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


__all__ = [
    "ChunkerPositionAdapter",
]
