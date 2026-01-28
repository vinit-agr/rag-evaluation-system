"""Chunk-level synthetic data generation.

This module provides synthetic data generation with chunk-level ground truth,
where the LLM generates queries AND identifies relevant chunk IDs together.
"""

from rag_evaluation_system.synthetic_datagen.chunk_level.generator import (
    ChunkLevelSyntheticDatasetGenerator,
    GeneratedQAPair,
)

__all__ = [
    "ChunkLevelSyntheticDatasetGenerator",
    "GeneratedQAPair",
]
