"""Recursive character text splitter implementation.

This module provides a chunker that recursively splits text using a hierarchy
of separators, attempting to keep semantically related content together.
"""

from typing import ClassVar

from rag_evaluation_system.chunkers.base import Chunker, PositionAwareChunker
from rag_evaluation_system.types.chunks import PositionAwareChunk
from rag_evaluation_system.types.documents import Document
from rag_evaluation_system.utils.hashing import generate_pa_chunk_id


class RecursiveCharacterChunker(Chunker, PositionAwareChunker):
    """A chunker that recursively splits text using a hierarchy of separators.

    This implementation tries to keep semantically related content together by:
    1. First attempting to split on paragraph breaks (double newlines)
    2. Then on line breaks
    3. Then on sentence boundaries
    4. Then on word boundaries
    5. Finally, character-by-character as a last resort

    The algorithm merges small chunks together until they reach the target size,
    and adds overlap from the previous chunk for context continuity.

    This chunker implements BOTH Chunker and PositionAwareChunker interfaces,
    making it suitable for both chunk-level and token-level evaluation.

    Attributes:
        chunk_size: Target size for each chunk in characters.
        chunk_overlap: Number of characters to overlap between chunks.
        separators: List of separators to try, in order of preference.

    Example:
        >>> chunker = RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50)
        >>> chunks = chunker.chunk(document_text)
        >>> # Or with position tracking:
        >>> pa_chunks = chunker.chunk_with_positions(document)
    """

    DEFAULT_SEPARATORS: ClassVar[list[str]] = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        """Initialize the recursive character chunker.

        Args:
            chunk_size: Target size for each chunk in characters. Default: 1000.
            chunk_overlap: Number of characters to overlap between chunks.
                Must be less than chunk_size. Default: 200.
            separators: List of separators to try in order. If None, uses
                default separators: ["\\n\\n", "\\n", ". ", " ", ""].

        Raises:
            ValueError: If chunk_overlap >= chunk_size.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators if separators is not None else self.DEFAULT_SEPARATORS.copy()

    @property
    def name(self) -> str:
        """Return a descriptive name for this chunker configuration.

        Returns:
            A string identifier including chunk size and overlap.
        """
        return f"recursive-character(size={self._chunk_size}, overlap={self._chunk_overlap})"

    @property
    def chunk_size(self) -> int:
        """Return the target chunk size."""
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        """Return the chunk overlap size."""
        return self._chunk_overlap

    @property
    def separators(self) -> list[str]:
        """Return the list of separators."""
        return self._separators.copy()

    def chunk(self, text: str) -> list[str]:
        """Split text into chunks using recursive character splitting.

        Args:
            text: The full text to chunk.

        Returns:
            List of chunk text strings.
        """
        if not text:
            return []

        return self._split_text(text, self._separators)

    def chunk_with_positions(self, doc: Document) -> list[PositionAwareChunk]:
        """Split document into position-aware chunks.

        Args:
            doc: The document to chunk.

        Returns:
            List of PositionAwareChunk objects with character positions.
        """
        if not doc.content:
            return []

        chunks_with_positions = self._split_text_with_positions(
            doc.content, self._separators, offset=0
        )

        result: list[PositionAwareChunk] = []
        for chunk_text, start, end in chunks_with_positions:
            result.append(
                PositionAwareChunk(
                    id=generate_pa_chunk_id(chunk_text),
                    content=chunk_text,
                    doc_id=doc.id,
                    start=start,
                    end=end,
                )
            )

        return result

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the given separators.

        Args:
            text: The text to split.
            separators: Remaining separators to try.

        Returns:
            List of chunk strings.
        """
        final_chunks: list[str] = []

        # Find the appropriate separator
        separator = separators[-1]  # Default to last (empty string = char split)
        new_separators: list[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        # Split by the chosen separator
        splits = self._split_by_separator(text, separator)

        # Merge splits into chunks of appropriate size
        good_splits: list[str] = []
        for split in splits:
            if len(split) < self._chunk_size:
                good_splits.append(split)
            else:
                # This split is too large, need to process accumulated good_splits first
                if good_splits:
                    merged = self._merge_splits(good_splits)
                    final_chunks.extend(merged)
                    good_splits = []

                # Now handle the large split
                if not new_separators:
                    # No more separators, just add as-is (will be oversized)
                    final_chunks.append(split)
                else:
                    # Recursively split with remaining separators
                    sub_chunks = self._split_text(split, new_separators)
                    final_chunks.extend(sub_chunks)

        # Don't forget remaining good_splits
        if good_splits:
            merged = self._merge_splits(good_splits)
            final_chunks.extend(merged)

        return final_chunks

    def _split_text_with_positions(
        self, text: str, separators: list[str], offset: int
    ) -> list[tuple[str, int, int]]:
        """Recursively split text with position tracking.

        Args:
            text: The text to split.
            separators: Remaining separators to try.
            offset: Character offset in the original document.

        Returns:
            List of tuples (chunk_text, start_position, end_position).
        """
        final_chunks: list[tuple[str, int, int]] = []

        # Find the appropriate separator
        separator = separators[-1]
        new_separators: list[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        # Split by the chosen separator with positions
        splits_with_pos = self._split_by_separator_with_positions(text, separator, offset)

        # Merge splits into chunks of appropriate size
        good_splits: list[tuple[str, int, int]] = []
        for split_text, start, end in splits_with_pos:
            if len(split_text) < self._chunk_size:
                good_splits.append((split_text, start, end))
            else:
                # Process accumulated good_splits first
                if good_splits:
                    merged = self._merge_splits_with_positions(good_splits)
                    final_chunks.extend(merged)
                    good_splits = []

                # Handle the large split
                if not new_separators:
                    final_chunks.append((split_text, start, end))
                else:
                    sub_chunks = self._split_text_with_positions(split_text, new_separators, start)
                    final_chunks.extend(sub_chunks)

        # Don't forget remaining good_splits
        if good_splits:
            merged = self._merge_splits_with_positions(good_splits)
            final_chunks.extend(merged)

        return final_chunks

    def _split_by_separator(self, text: str, separator: str) -> list[str]:
        """Split text by a separator, keeping non-empty results.

        Args:
            text: The text to split.
            separator: The separator to split on.

        Returns:
            List of non-empty splits. If separator is empty, splits into characters.
        """
        if separator == "":
            # Character-level split
            return list(text)

        parts = text.split(separator)
        # Re-attach separator to the end of each part except the last
        result: list[str] = []
        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                # Add separator back to non-final parts
                result.append(part + separator)
            elif part:  # Only add final part if non-empty
                result.append(part)

        return result

    def _split_by_separator_with_positions(
        self, text: str, separator: str, offset: int
    ) -> list[tuple[str, int, int]]:
        """Split text by separator while tracking positions.

        Args:
            text: The text to split.
            separator: The separator to split on.
            offset: Starting offset for positions.

        Returns:
            List of tuples (split_text, start_position, end_position).
        """
        if separator == "":
            # Character-level split
            return [(char, offset + i, offset + i + 1) for i, char in enumerate(text)]

        result: list[tuple[str, int, int]] = []
        current_pos = 0

        parts = text.split(separator)
        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                # Non-final part: include separator
                chunk = part + separator
                start = offset + current_pos
                end = start + len(chunk)
                if chunk:  # Only include non-empty
                    result.append((chunk, start, end))
                current_pos += len(chunk)
            elif part:
                # Final part, only if non-empty
                start = offset + current_pos
                end = start + len(part)
                result.append((part, start, end))

        return result

    def _merge_splits(self, splits: list[str]) -> list[str]:
        """Merge small splits into chunks of target size with overlap.

        Args:
            splits: List of text splits to merge.

        Returns:
            List of merged chunks.
        """
        if not splits:
            return []

        merged_chunks: list[str] = []
        current_chunk_parts: list[str] = []
        current_length = 0

        for split in splits:
            split_len = len(split)

            # Check if adding this split would exceed chunk_size
            if current_length + split_len > self._chunk_size and current_chunk_parts:
                # Finalize current chunk
                chunk = "".join(current_chunk_parts)
                merged_chunks.append(chunk)

                # Start new chunk with overlap from previous
                overlap_parts = self._get_overlap_parts(current_chunk_parts)
                current_chunk_parts = overlap_parts
                current_length = sum(len(p) for p in current_chunk_parts)

            current_chunk_parts.append(split)
            current_length += split_len

        # Don't forget the last chunk
        if current_chunk_parts:
            chunk = "".join(current_chunk_parts)
            merged_chunks.append(chunk)

        return merged_chunks

    def _merge_splits_with_positions(
        self, splits: list[tuple[str, int, int]]
    ) -> list[tuple[str, int, int]]:
        """Merge small splits with position tracking.

        Args:
            splits: List of (text, start, end) tuples to merge.

        Returns:
            List of merged (text, start, end) tuples.
        """
        if not splits:
            return []

        merged_chunks: list[tuple[str, int, int]] = []
        current_parts: list[tuple[str, int, int]] = []
        current_length = 0

        for split_text, start, end in splits:
            split_len = len(split_text)

            if current_length + split_len > self._chunk_size and current_parts:
                # Finalize current chunk
                chunk_text = "".join(p[0] for p in current_parts)
                chunk_start = current_parts[0][1]
                chunk_end = current_parts[-1][2]
                merged_chunks.append((chunk_text, chunk_start, chunk_end))

                # Start new chunk with overlap
                overlap_parts = self._get_overlap_parts_with_positions(current_parts)
                current_parts = overlap_parts
                current_length = sum(len(p[0]) for p in current_parts)

            current_parts.append((split_text, start, end))
            current_length += split_len

        # Don't forget the last chunk
        if current_parts:
            chunk_text = "".join(p[0] for p in current_parts)
            chunk_start = current_parts[0][1]
            chunk_end = current_parts[-1][2]
            merged_chunks.append((chunk_text, chunk_start, chunk_end))

        return merged_chunks

    def _get_overlap_parts(self, parts: list[str]) -> list[str]:
        """Get parts from the end that fit within the overlap size.

        Args:
            parts: List of text parts.

        Returns:
            List of parts that should form the overlap for the next chunk.
        """
        if self._chunk_overlap == 0:
            return []

        overlap_parts: list[str] = []
        overlap_length = 0

        # Work backwards through parts
        for part in reversed(parts):
            if overlap_length + len(part) <= self._chunk_overlap:
                overlap_parts.insert(0, part)
                overlap_length += len(part)
            else:
                break

        return overlap_parts

    def _get_overlap_parts_with_positions(
        self, parts: list[tuple[str, int, int]]
    ) -> list[tuple[str, int, int]]:
        """Get parts from the end that fit within the overlap size, with positions.

        Args:
            parts: List of (text, start, end) tuples.

        Returns:
            List of tuples that should form the overlap for the next chunk.
        """
        if self._chunk_overlap == 0:
            return []

        overlap_parts: list[tuple[str, int, int]] = []
        overlap_length = 0

        for part in reversed(parts):
            if overlap_length + len(part[0]) <= self._chunk_overlap:
                overlap_parts.insert(0, part)
                overlap_length += len(part[0])
            else:
                break

        return overlap_parts


__all__ = [
    "RecursiveCharacterChunker",
]
