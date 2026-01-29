"""Hashing utilities for generating chunk IDs.

This module provides functions for generating deterministic, content-based
identifiers for chunks. Using content hashes ensures:
- Deterministic: same content always produces same ID
- Deduplication: identical chunks have identical IDs
- Stable: ID doesn't change based on processing order
"""

import hashlib

from rag_evaluation_system.types.primitives import ChunkId, PositionAwareChunkId


def generate_chunk_id(content: str) -> ChunkId:
    """Generate a standard chunk ID from content.

    Format: "chunk_" + first 12 chars of SHA256 hash.

    Args:
        content: The text content to hash.

    Returns:
        A ChunkId with the format "chunk_" + 12-character hash.

    Example:
        >>> generate_chunk_id("Hello, World!")
        ChunkId('chunk_dffd6021bb2b')
    """
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
    return ChunkId(f"chunk_{content_hash}")


def generate_pa_chunk_id(content: str) -> PositionAwareChunkId:
    """Generate a position-aware chunk ID from content.

    Format: "pa_chunk_" + first 12 chars of SHA256 hash.

    The "pa_" prefix distinguishes these from regular chunk IDs,
    making it immediately clear when working with position-aware data.

    Args:
        content: The text content to hash.

    Returns:
        A PositionAwareChunkId with the format "pa_chunk_" + 12-character hash.

    Example:
        >>> generate_pa_chunk_id("Hello, World!")
        PositionAwareChunkId('pa_chunk_dffd6021bb2b')
    """
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
    return PositionAwareChunkId(f"pa_chunk_{content_hash}")


__all__ = [
    "generate_chunk_id",
    "generate_pa_chunk_id",
]
