"""Ground truth models for the RAG evaluation system.

This module defines data structures for representing ground truth data
in both chunk-level and token-level evaluation paradigms, as well as
the corresponding LangSmith dataset example schemas.
"""

from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict

from rag_evaluation_system.types.chunks import CharacterSpan
from rag_evaluation_system.types.primitives import ChunkId, QueryText
from rag_evaluation_system.types.queries import Query


class ChunkLevelGroundTruth(BaseModel):
    """Ground truth data for a single query in chunk-level evaluation.

    Maps a query to the list of chunk IDs that are considered relevant.
    Used to measure retrieval performance at the chunk level.

    Attributes:
        query: The query this ground truth is for.
        relevant_chunk_ids: List of chunk IDs that are relevant to this query.
            Format: ["chunk_a3f2b1c8d9e0", "chunk_7d9e4f2a1b3c", ...]
    """

    model_config = ConfigDict(frozen=True)

    query: Query
    relevant_chunk_ids: list[ChunkId]


class TokenLevelGroundTruth(BaseModel):
    """Ground truth data for a single query in token-level evaluation.

    Maps a query to the list of character spans that contain relevant content.
    These spans are extracted directly from documents during synthetic data
    generation - NO chunking is involved.

    Attributes:
        query: The query this ground truth is for.
        relevant_spans: List of CharacterSpan objects representing the exact
            excerpts from documents that answer the query.

    Note:
        Ground truth is chunker-independent. The same ground truth dataset
        can be used to evaluate ANY chunking strategy.
    """

    model_config = ConfigDict(frozen=True)

    query: Query
    relevant_spans: list[CharacterSpan]


# =============================================================================
# LangSmith Dataset Example TypedDicts
# =============================================================================
# These TypedDicts define the schema for storing/retrieving data from LangSmith.
# They follow LangSmith's inputs/outputs/metadata convention.


class ChunkLevelInputs(TypedDict):
    """Input schema for chunk-level dataset examples."""

    query: QueryText


class ChunkLevelOutputs(TypedDict):
    """Output schema for chunk-level dataset examples."""

    relevant_chunk_ids: list[ChunkId]


class ChunkLevelDatasetExample(TypedDict):
    """LangSmith dataset example schema for chunk-level evaluation.

    This is the format used when storing/retrieving data from LangSmith.
    Follows LangSmith's inputs/outputs/metadata convention.

    Example:
        {
            "inputs": {"query": "What is RAG?"},
            "outputs": {"relevant_chunk_ids": ["chunk_xxx", "chunk_yyy"]},
            "metadata": {"source_docs": ["rag.md"], "generation_model": "gpt-4"}
        }
    """

    inputs: ChunkLevelInputs
    outputs: ChunkLevelOutputs
    metadata: dict[str, Any]


class CharacterSpanDict(TypedDict):
    """Dictionary representation of a CharacterSpan for LangSmith storage."""

    doc_id: str
    start: int
    end: int
    text: str


class TokenLevelInputs(TypedDict):
    """Input schema for token-level dataset examples."""

    query: QueryText


class TokenLevelOutputs(TypedDict):
    """Output schema for token-level dataset examples."""

    relevant_spans: list[CharacterSpanDict]


class TokenLevelDatasetExample(TypedDict):
    """LangSmith dataset example schema for token-level evaluation.

    This is the format used when storing/retrieving data from LangSmith.
    Stores full character span data including text for convenience.

    Example:
        {
            "inputs": {"query": "What is RAG?"},
            "outputs": {
                "relevant_spans": [
                    {"doc_id": "rag.md", "start": 100, "end": 200, "text": "..."}
                ]
            },
            "metadata": {"source_docs": ["rag.md"], "generation_model": "gpt-4"}
        }
    """

    inputs: TokenLevelInputs
    outputs: TokenLevelOutputs
    metadata: dict[str, Any]


__all__ = [
    "CharacterSpanDict",
    "ChunkLevelDatasetExample",
    "ChunkLevelGroundTruth",
    "ChunkLevelInputs",
    "ChunkLevelOutputs",
    "TokenLevelDatasetExample",
    "TokenLevelGroundTruth",
    "TokenLevelInputs",
    "TokenLevelOutputs",
]
