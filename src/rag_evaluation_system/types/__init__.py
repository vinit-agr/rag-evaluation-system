"""Type definitions for the RAG evaluation system.

This module exports all type definitions used throughout the framework,
providing strong typing and clear semantics for all data structures.
"""

from rag_evaluation_system.types.chunks import (
    CharacterSpan,
    Chunk,
    PositionAwareChunk,
)
from rag_evaluation_system.types.documents import (
    Corpus,
    Document,
)
from rag_evaluation_system.types.ground_truth import (
    CharacterSpanDict,
    ChunkLevelDatasetExample,
    ChunkLevelGroundTruth,
    ChunkLevelInputs,
    ChunkLevelOutputs,
    TokenLevelDatasetExample,
    TokenLevelGroundTruth,
    TokenLevelInputs,
    TokenLevelOutputs,
)
from rag_evaluation_system.types.primitives import (
    ChunkId,
    DocumentId,
    EvaluationType,
    PositionAwareChunkId,
    QueryId,
    QueryText,
)
from rag_evaluation_system.types.queries import (
    Query,
)
from rag_evaluation_system.types.results import (
    ChunkLevelRunOutput,
    EvaluationResult,
    TokenLevelRunOutput,
)

__all__ = [
    "CharacterSpan",
    "CharacterSpanDict",
    "Chunk",
    "ChunkId",
    "ChunkLevelDatasetExample",
    "ChunkLevelGroundTruth",
    "ChunkLevelInputs",
    "ChunkLevelOutputs",
    "ChunkLevelRunOutput",
    "Corpus",
    "Document",
    "DocumentId",
    "EvaluationResult",
    "EvaluationType",
    "PositionAwareChunk",
    "PositionAwareChunkId",
    "Query",
    "QueryId",
    "QueryText",
    "TokenLevelDatasetExample",
    "TokenLevelGroundTruth",
    "TokenLevelInputs",
    "TokenLevelOutputs",
    "TokenLevelRunOutput",
]
