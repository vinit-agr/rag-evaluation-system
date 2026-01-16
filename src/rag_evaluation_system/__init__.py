"""RAG Evaluation System - A framework for evaluating RAG retrieval pipelines.

This package provides tools for evaluating Retrieval-Augmented Generation (RAG)
pipelines at both chunk-level and token-level granularity.

Two Evaluation Paradigms:
    - Chunk-Level: Ground truth is a list of chunk IDs. Simpler but ties
      evaluation to a specific chunking strategy.
    - Token-Level (Recommended): Ground truth is character spans. Chunker-independent,
      enabling fair comparison across different chunking strategies.

Example:
    >>> from rag_evaluation_system import (
    ...     Corpus,
    ...     Document,
    ...     RecursiveCharacterChunker,
    ...     ChunkLevelEvaluation,
    ...     ChunkRecall,
    ... )
    >>> corpus = Corpus(
    ...     documents=[
    ...         Document(id="doc1", content="Some content..."),
    ...     ]
    ... )
    >>> chunker = RecursiveCharacterChunker(chunk_size=500)
"""

__version__ = "0.1.0"

from rag_evaluation_system.chunkers import (
    Chunker,
    ChunkerPositionAdapter,
    PositionAwareChunker,
    RecursiveCharacterChunker,
)
from rag_evaluation_system.embedders import Embedder
from rag_evaluation_system.evaluation import (
    ChunkLevelEvaluation,
    TokenLevelEvaluation,
)
from rag_evaluation_system.evaluation.metrics import (
    ChunkF1,
    ChunkPrecision,
    ChunkRecall,
    SpanIoU,
    SpanPrecision,
    SpanRecall,
)
from rag_evaluation_system.rerankers import Reranker
from rag_evaluation_system.synthetic_datagen import (
    ChunkLevelSyntheticDatasetGenerator,
    TokenLevelSyntheticDatasetGenerator,
)
from rag_evaluation_system.types import (
    CharacterSpan,
    Chunk,
    ChunkId,
    ChunkLevelGroundTruth,
    Corpus,
    Document,
    DocumentId,
    EvaluationResult,
    PositionAwareChunk,
    PositionAwareChunkId,
    Query,
    QueryId,
    QueryText,
    TokenLevelGroundTruth,
)
from rag_evaluation_system.vector_stores import VectorStore

__all__ = [
    "CharacterSpan",
    "Chunk",
    "ChunkF1",
    "ChunkId",
    "ChunkLevelEvaluation",
    "ChunkLevelGroundTruth",
    "ChunkLevelSyntheticDatasetGenerator",
    "ChunkPrecision",
    "ChunkRecall",
    "Chunker",
    "ChunkerPositionAdapter",
    "Corpus",
    "Document",
    "DocumentId",
    "Embedder",
    "EvaluationResult",
    "PositionAwareChunk",
    "PositionAwareChunkId",
    "PositionAwareChunker",
    "Query",
    "QueryId",
    "QueryText",
    "RecursiveCharacterChunker",
    "Reranker",
    "SpanIoU",
    "SpanPrecision",
    "SpanRecall",
    "TokenLevelEvaluation",
    "TokenLevelGroundTruth",
    "TokenLevelSyntheticDatasetGenerator",
    "VectorStore",
    "__version__",
]

# =============================================================================
# Conditional imports for optional dependencies
# =============================================================================

# OpenAI Embedder (requires 'openai' extra)
try:
    from rag_evaluation_system.embedders import OpenAIEmbedder as OpenAIEmbedder

    __all__.append("OpenAIEmbedder")
except ImportError:
    pass

# SentenceTransformer Embedder (requires 'sentence-transformers' extra)
try:
    from rag_evaluation_system.embedders import (
        SentenceTransformerEmbedder as SentenceTransformerEmbedder,
    )

    __all__.append("SentenceTransformerEmbedder")
except ImportError:
    pass

# Chroma Vector Store (requires 'chroma' extra)
try:
    from rag_evaluation_system.vector_stores import (
        ChromaVectorStore as ChromaVectorStore,
    )

    __all__.append("ChromaVectorStore")
except ImportError:
    pass

# Cohere Reranker (requires 'cohere' extra)
try:
    from rag_evaluation_system.rerankers import CohereReranker as CohereReranker

    __all__.append("CohereReranker")
except ImportError:
    pass
