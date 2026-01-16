"""Synthetic data generation for RAG evaluation.

This module provides synthetic data generators for both chunk-level and
token-level evaluation paradigms.

Chunk-Level:
    Ground truth is a list of chunk IDs. Requires a chunker at generation time.
    Use ChunkLevelSyntheticDatasetGenerator.

Token-Level (Recommended):
    Ground truth is character spans (doc_id, start, end, text). No chunker needed.
    Allows fair comparison of different chunking strategies.
    Use TokenLevelSyntheticDatasetGenerator.
"""

from rag_evaluation_system.synthetic_datagen.base import SyntheticDatasetGenerator
from rag_evaluation_system.synthetic_datagen.chunk_level import (
    ChunkLevelSyntheticDatasetGenerator,
    GeneratedQAPair,
)
from rag_evaluation_system.synthetic_datagen.token_level import (
    ExtractedExcerpt,
    GeneratedQAWithExcerpts,
    TokenLevelSyntheticDatasetGenerator,
)

__all__ = [
    "ChunkLevelSyntheticDatasetGenerator",
    "ExtractedExcerpt",
    "GeneratedQAPair",
    "GeneratedQAWithExcerpts",
    "SyntheticDatasetGenerator",
    "TokenLevelSyntheticDatasetGenerator",
]
