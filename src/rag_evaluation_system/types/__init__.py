"""Type definitions for the RAG evaluation system."""
from .chunks import CharacterSpan, Chunk, PositionAwareChunk
from .documents import Corpus, Document
from .ground_truth import (
    ChunkLevelDatasetExample,
    ChunkLevelGroundTruth,
    TokenLevelDatasetExample,
    TokenLevelGroundTruth,
)
from .primitives import (
    ChunkId,
    DocumentId,
    EvaluationType,
    PositionAwareChunkId,
    QueryId,
    QueryText,
)
from .queries import Query
from .results import EvaluationResult, ChunkLevelRunOutput, TokenLevelRunOutput

__all__ = [
    "DocumentId",
    "QueryId",
    "QueryText",
    "ChunkId",
    "PositionAwareChunkId",
    "EvaluationType",
    "Document",
    "Corpus",
    "CharacterSpan",
    "Chunk",
    "PositionAwareChunk",
    "Query",
    "ChunkLevelGroundTruth",
    "TokenLevelGroundTruth",
    "ChunkLevelDatasetExample",
    "TokenLevelDatasetExample",
    "EvaluationResult",
    "ChunkLevelRunOutput",
    "TokenLevelRunOutput",
]
