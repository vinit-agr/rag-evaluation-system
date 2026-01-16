"""Result types for evaluation runs."""
from typing import Any

from pydantic import BaseModel

from .chunks import CharacterSpan
from .primitives import ChunkId


class EvaluationResult(BaseModel):
    """Results from an evaluation run."""

    metrics: dict[str, float]
    experiment_url: str | None = None
    raw_results: Any = None


class ChunkLevelRunOutput(BaseModel):
    """Output from retrieval pipeline for chunk-level evaluation."""

    retrieved_chunk_ids: list[ChunkId]


class TokenLevelRunOutput(BaseModel):
    """Output from retrieval pipeline for token-level evaluation."""

    retrieved_spans: list[CharacterSpan]
