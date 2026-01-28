"""Chunker interfaces and implementations.

This module provides:
- Chunker: Simple interface returning plain text chunks
- PositionAwareChunker: Interface returning chunks with position tracking
- ChunkerPositionAdapter: Adapter to wrap any Chunker as position-aware
- RecursiveCharacterChunker: Implementation using recursive separator-based splitting
"""

from rag_evaluation_system.chunkers.adapter import ChunkerPositionAdapter
from rag_evaluation_system.chunkers.base import Chunker, PositionAwareChunker
from rag_evaluation_system.chunkers.recursive_character import RecursiveCharacterChunker

__all__ = [
    "Chunker",
    "ChunkerPositionAdapter",
    "PositionAwareChunker",
    "RecursiveCharacterChunker",
]
