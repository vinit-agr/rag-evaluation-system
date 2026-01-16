"""RAG Evaluation System - Comprehensive RAG retrieval evaluation."""
from rag_evaluation_system.chunkers import (
    Chunker,
    ChunkerPositionAdapter,
    PositionAwareChunker,
    RecursiveCharacterChunker,
)
from rag_evaluation_system.evaluation import ChunkLevelEvaluation, TokenLevelEvaluation
from rag_evaluation_system.rerankers import Reranker
from rag_evaluation_system.synthetic_datagen import (
    ChunkLevelSyntheticDatasetGenerator,
    TokenLevelSyntheticDatasetGenerator,
)
from rag_evaluation_system.types import (
    CharacterSpan,
    Chunk,
    ChunkLevelGroundTruth,
    Corpus,
    Document,
    EvaluationResult,
    PositionAwareChunk,
    Query,
    TokenLevelGroundTruth,
)
from rag_evaluation_system.vector_stores import VectorStore
from .embedders import Embedder

try:
    from rag_evaluation_system.embedders.openai import OpenAIEmbedder
except ImportError:  # pragma: no cover
    OpenAIEmbedder = None  # type: ignore[assignment]

try:
    from rag_evaluation_system.vector_stores.chroma import ChromaVectorStore
except ImportError:  # pragma: no cover
    ChromaVectorStore = None  # type: ignore[assignment]

try:
    from rag_evaluation_system.rerankers.cohere import CohereReranker
except ImportError:  # pragma: no cover
    CohereReranker = None  # type: ignore[assignment]

__version__ = "0.1.0"

__all__ = [
    "Document",
    "Corpus",
    "Chunk",
    "PositionAwareChunk",
    "CharacterSpan",
    "Query",
    "ChunkLevelGroundTruth",
    "TokenLevelGroundTruth",
    "EvaluationResult",
    "Chunker",
    "PositionAwareChunker",
    "ChunkerPositionAdapter",
    "RecursiveCharacterChunker",
    "Embedder",
    "VectorStore",
    "Reranker",
    "ChunkLevelSyntheticDatasetGenerator",
    "TokenLevelSyntheticDatasetGenerator",
    "ChunkLevelEvaluation",
    "TokenLevelEvaluation",
    "OpenAIEmbedder",
    "ChromaVectorStore",
    "CohereReranker",
]
