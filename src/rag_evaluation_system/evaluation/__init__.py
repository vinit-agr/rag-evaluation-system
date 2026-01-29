"""Evaluation orchestrators and metrics for RAG pipelines.

This module provides evaluation orchestrators for both chunk-level and token-level
evaluation paradigms, as well as metrics for measuring retrieval performance.

Orchestrators:
    - ChunkLevelEvaluation: Evaluates retrieval using chunk ID matching
    - TokenLevelEvaluation: Evaluates retrieval using character span overlap

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

Example:
    >>> from rag_evaluation_system.evaluation import (
    ...     ChunkLevelEvaluation,
    ...     TokenLevelEvaluation,
    ...     ChunkRecall,
    ...     SpanRecall,
    ... )
    >>> # Chunk-level evaluation
    >>> eval = ChunkLevelEvaluation(corpus, "my-dataset")
    >>> result = eval.run(chunker, embedder)
    >>> print(result.metrics["chunk_recall"])
    0.85
"""

from rag_evaluation_system.evaluation.chunk_level import ChunkLevelEvaluation
from rag_evaluation_system.evaluation.metrics import (
    ChunkF1,
    ChunkLevelMetric,
    ChunkPrecision,
    ChunkRecall,
    SpanIoU,
    SpanPrecision,
    SpanRecall,
    TokenLevelMetric,
    calculate_overlap,
    merge_overlapping_spans,
)
from rag_evaluation_system.evaluation.token_level import TokenLevelEvaluation

__all__ = [
    "ChunkF1",
    "ChunkLevelEvaluation",
    "ChunkLevelMetric",
    "ChunkPrecision",
    "ChunkRecall",
    "SpanIoU",
    "SpanPrecision",
    "SpanRecall",
    "TokenLevelEvaluation",
    "TokenLevelMetric",
    "calculate_overlap",
    "merge_overlapping_spans",
]
