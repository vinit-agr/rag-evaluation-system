"""Tests for RecursiveCharacterChunker."""
from rag_evaluation_system.chunkers.recursive_character import RecursiveCharacterChunker


def test_recursive_character_chunker_max_size():
    text = "a" * 250
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(text)

    assert all(len(chunk) <= 100 for chunk in chunks)
    assert len(chunks) > 1
