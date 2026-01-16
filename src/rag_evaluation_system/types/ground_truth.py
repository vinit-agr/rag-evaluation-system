"""Ground truth types for chunk-level and token-level evaluation."""
from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict

from .chunks import CharacterSpan
from .primitives import ChunkId, QueryText
from .queries import Query


class ChunkLevelGroundTruth(BaseModel):
    """Ground truth for chunk-level evaluation."""

    model_config = ConfigDict(frozen=True)

    query: Query
    relevant_chunk_ids: list[ChunkId]


class TokenLevelGroundTruth(BaseModel):
    """Ground truth for token-level evaluation."""

    model_config = ConfigDict(frozen=True)

    query: Query
    relevant_spans: list[CharacterSpan]


class ChunkLevelDatasetExample(TypedDict):
    """LangSmith dataset example schema for chunk-level evaluation."""

    inputs: dict[str, QueryText]
    outputs: dict[str, list[ChunkId]]
    metadata: dict[str, Any]


class TokenLevelDatasetExample(TypedDict):
    """LangSmith dataset example schema for token-level evaluation."""

    inputs: dict[str, QueryText]
    outputs: dict[str, list[CharacterSpan]]
    metadata: dict[str, Any]
