"""Evaluation result models for the RAG evaluation system.

This module defines data structures for representing evaluation results
and pipeline outputs in both chunk-level and token-level evaluation.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict

from rag_evaluation_system.types.chunks import CharacterSpan
from rag_evaluation_system.types.primitives import ChunkId


class EvaluationResult(BaseModel):
    """Results from an evaluation run.

    Contains computed metrics, optional experiment URL for LangSmith,
    and raw results for further analysis.

    Attributes:
        metrics: Dictionary mapping metric names to computed values.
            Examples: {"chunk_recall": 0.85, "chunk_precision": 0.72}
        experiment_url: URL to the LangSmith experiment page, if available.
        raw_results: Raw results object from LangSmith for further analysis.
    """

    model_config = ConfigDict(frozen=True)

    metrics: dict[str, float]
    experiment_url: str | None = None
    raw_results: Any = None


class ChunkLevelRunOutput(BaseModel):
    """Output from the retrieval pipeline for chunk-level evaluation.

    This is what the retrieval function returns for each query in
    chunk-level evaluation.

    Attributes:
        retrieved_chunk_ids: List of chunk IDs retrieved for the query.
            Format: ["chunk_xxx", "chunk_yyy", ...]
    """

    model_config = ConfigDict(frozen=True)

    retrieved_chunk_ids: list[ChunkId]


class TokenLevelRunOutput(BaseModel):
    """Output from the retrieval pipeline for token-level evaluation.

    This is what the retrieval function returns for each query in
    token-level evaluation. The retrieved chunks are position-aware
    so we can compute span overlap with ground truth.

    Attributes:
        retrieved_spans: List of CharacterSpans representing the retrieved
            content positions. Converted from PositionAwareChunks.
    """

    model_config = ConfigDict(frozen=True)

    retrieved_spans: list[CharacterSpan]


__all__ = [
    "ChunkLevelRunOutput",
    "EvaluationResult",
    "TokenLevelRunOutput",
]
