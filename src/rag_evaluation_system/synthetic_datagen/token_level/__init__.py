"""Token-level synthetic data generation.

This module provides synthetic data generation with character span ground truth,
which is chunker-independent and allows fair comparison of different chunking
strategies against the same ground truth.
"""

from rag_evaluation_system.synthetic_datagen.token_level.generator import (
    ExtractedExcerpt,
    GeneratedQAWithExcerpts,
    TokenLevelSyntheticDatasetGenerator,
)

__all__ = [
    "ExtractedExcerpt",
    "GeneratedQAWithExcerpts",
    "TokenLevelSyntheticDatasetGenerator",
]
