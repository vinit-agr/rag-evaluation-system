"""Tests for chunker adapter."""
from rag_evaluation_system.chunkers.adapter import ChunkerPositionAdapter
from rag_evaluation_system.chunkers.base import Chunker
from rag_evaluation_system.types import Document, DocumentId


class SimpleChunker(Chunker):
    @property
    def name(self) -> str:
        return "Simple"

    def chunk(self, text: str) -> list[str]:
        return [text[:5], text[5:]]


def test_chunker_adapter_positions():
    doc = Document(id=DocumentId("doc1"), content="hello_world")
    adapter = ChunkerPositionAdapter(SimpleChunker())
    chunks = adapter.chunk_with_positions(doc)

    assert len(chunks) == 2
    assert chunks[0].start == 0
    assert chunks[0].end == 5
    assert chunks[1].start == 5
    assert chunks[1].end == len(doc.content)
