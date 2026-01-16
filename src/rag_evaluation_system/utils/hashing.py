"""Utilities for generating chunk IDs."""
import hashlib

from rag_evaluation_system.types import ChunkId, PositionAwareChunkId


def generate_chunk_id(content: str) -> ChunkId:
    """Generate a standard chunk ID from content."""
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return ChunkId(f"chunk_{content_hash}")


def generate_pa_chunk_id(content: str) -> PositionAwareChunkId:
    """Generate a position-aware chunk ID from content."""
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return PositionAwareChunkId(f"pa_chunk_{content_hash}")
