"""Metrics for RAG evaluation.

This module provides metrics for both chunk-level and token-level evaluation:

Chunk-Level Metrics:
    - ChunkRecall: Proportion of ground truth chunks retrieved
    - ChunkPrecision: Proportion of retrieved chunks that are relevant
    - ChunkF1: Harmonic mean of precision and recall

Token-Level Metrics:
    - SpanRecall: Proportion of ground truth characters retrieved
    - SpanPrecision: Proportion of retrieved characters that are relevant
    - SpanIoU: Intersection over Union of character spans

Base Classes:
    - ChunkLevelMetric: ABC for chunk-level metrics
    - TokenLevelMetric: ABC for token-level metrics

Utilities:
    - merge_overlapping_spans: Merge overlapping character spans
    - calculate_overlap: Calculate character overlap between span collections
"""

from rag_evaluation_system.evaluation.metrics.base import (
    ChunkLevelMetric,
    TokenLevelMetric,
)
from rag_evaluation_system.evaluation.metrics.chunk_level import (
    ChunkF1,
    ChunkPrecision,
    ChunkRecall,
)
from rag_evaluation_system.evaluation.metrics.token_level import (
    SpanIoU,
    SpanPrecision,
    SpanRecall,
    calculate_overlap,
    merge_overlapping_spans,
)

__all__ = [
    "ChunkF1",
    "ChunkLevelMetric",
    "ChunkPrecision",
    "ChunkRecall",
    "SpanIoU",
    "SpanPrecision",
    "SpanRecall",
    "TokenLevelMetric",
    "calculate_overlap",
    "merge_overlapping_spans",
]
