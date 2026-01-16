"""Recursive character text splitter implementation."""
from rag_evaluation_system.types import Document, PositionAwareChunk
from rag_evaluation_system.utils.hashing import generate_pa_chunk_id
from .base import Chunker, PositionAwareChunker


class RecursiveCharacterChunker(Chunker, PositionAwareChunker):
    """Recursive character text splitter."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or self.DEFAULT_SEPARATORS

    @property
    def name(self) -> str:
        return f"RecursiveCharacter(size={self._chunk_size}, overlap={self._chunk_overlap})"

    def chunk(self, text: str) -> list[str]:
        return self._merge_splits(self._recursive_split(text, self._separators))

    def chunk_with_positions(self, doc: Document) -> list[PositionAwareChunk]:
        splits = self._recursive_split_with_positions(doc.content, self._separators)
        merged = self._merge_splits_with_positions(splits)
        return [
            PositionAwareChunk(
                id=generate_pa_chunk_id(text),
                content=text,
                doc_id=doc.id,
                start=start,
                end=end,
            )
            for text, start, end in merged
        ]

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self._chunk_size:
            return [text]
        if not separators:
            return [text]

        separator = separators[0]
        if separator == "":
            return [text[i : i + self._chunk_size] for i in range(0, len(text), self._chunk_size)]

        pieces = text.split(separator)
        if len(pieces) == 1:
            return self._recursive_split(text, separators[1:])

        splits: list[str] = []
        for index, piece in enumerate(pieces):
            if index < len(pieces) - 1:
                piece = piece + separator
            if len(piece) > self._chunk_size:
                splits.extend(self._recursive_split(piece, separators[1:]))
            else:
                splits.append(piece)
        return splits

    def _merge_splits(self, splits: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        total = 0

        for split in splits:
            if total + len(split) > self._chunk_size and current:
                chunk = "".join(current)
                chunks.append(chunk)
                overlap_text = chunk[-self._chunk_overlap :] if self._chunk_overlap else ""
                current = [overlap_text] if overlap_text else []
                total = len(overlap_text)

            current.append(split)
            total += len(split)

        if current:
            chunks.append("".join(current))

        return chunks

    def _recursive_split_with_positions(
        self,
        text: str,
        separators: list[str],
        offset: int = 0,
    ) -> list[tuple[str, int, int]]:
        if len(text) <= self._chunk_size:
            return [(text, offset, offset + len(text))]
        if not separators:
            return [(text, offset, offset + len(text))]

        separator = separators[0]
        if separator == "":
            return [
                (text[i : i + self._chunk_size], offset + i, offset + min(i + self._chunk_size, len(text)))
                for i in range(0, len(text), self._chunk_size)
            ]

        pieces = text.split(separator)
        if len(pieces) == 1:
            return self._recursive_split_with_positions(text, separators[1:], offset)

        splits: list[tuple[str, int, int]] = []
        cursor = 0
        for index, piece in enumerate(pieces):
            part = piece + (separator if index < len(pieces) - 1 else "")
            start = offset + cursor
            end = start + len(part)
            if len(part) > self._chunk_size:
                splits.extend(self._recursive_split_with_positions(part, separators[1:], start))
            else:
                splits.append((part, start, end))
            cursor += len(part)
        return splits

    def _merge_splits_with_positions(
        self,
        splits: list[tuple[str, int, int]],
    ) -> list[tuple[str, int, int]]:
        chunks: list[tuple[str, int, int]] = []
        current_text = ""
        current_start: int | None = None
        current_end: int | None = None

        for split_text, split_start, split_end in splits:
            if current_start is None:
                current_start = split_start
                current_end = split_end
                current_text = split_text
                continue

            if len(current_text) + len(split_text) > self._chunk_size and current_text:
                chunks.append((current_text, current_start, current_end or current_start))

                overlap_len = min(self._chunk_overlap, len(current_text)) if self._chunk_overlap else 0
                if overlap_len:
                    current_start = (current_end or split_start) - overlap_len
                    current_text = current_text[-overlap_len:]
                    current_end = current_start + len(current_text)
                else:
                    current_text = ""
                    current_start = None
                    current_end = None

                if current_start is None:
                    current_start = split_start
                    current_text = split_text
                    current_end = split_end
                else:
                    current_text += split_text
                    current_end = split_end
            else:
                current_text += split_text
                current_end = split_end

        if current_start is not None and current_end is not None:
            chunks.append((current_text, current_start, current_end))

        return chunks
