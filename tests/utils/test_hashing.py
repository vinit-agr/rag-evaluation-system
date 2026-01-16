"""Tests for hashing utilities."""
from rag_evaluation_system.utils.hashing import generate_chunk_id, generate_pa_chunk_id


def test_generate_chunk_id_prefix():
    chunk_id = generate_chunk_id("hello")
    assert str(chunk_id).startswith("chunk_")


def test_generate_pa_chunk_id_prefix():
    chunk_id = generate_pa_chunk_id("hello")
    assert str(chunk_id).startswith("pa_chunk_")
