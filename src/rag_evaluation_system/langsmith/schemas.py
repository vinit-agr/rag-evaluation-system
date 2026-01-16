"""LangSmith dataset schemas."""
from typing import Any, TypedDict

from rag_evaluation_system.types import CharacterSpan, ChunkId, QueryText


class ChunkLevelDatasetSchema(TypedDict):
    inputs: dict[str, QueryText]
    outputs: dict[str, list[ChunkId]]
    metadata: dict[str, Any]


class TokenLevelDatasetSchema(TypedDict):
    inputs: dict[str, QueryText]
    outputs: dict[str, list[CharacterSpan]]
    metadata: dict[str, Any]
