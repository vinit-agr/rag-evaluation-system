"""Pytest fixtures for RAG evaluation system tests."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from rag_evaluation_system.chunkers import RecursiveCharacterChunker
from rag_evaluation_system.types import (
    CharacterSpan,
    Corpus,
    Document,
    DocumentId,
    PositionAwareChunk,
)

# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_document_content() -> str:
    """Return sample document content for testing."""
    return """# Sample Document

This is a sample document for testing the RAG evaluation system.

## Section 1

This section contains some important information about the topic.
It spans multiple lines and includes various details.

## Section 2

Another section with different content that can be used for
testing chunking and retrieval operations.

### Subsection 2.1

More detailed information in a subsection.
"""


@pytest.fixture
def sample_document_id() -> str:
    """Return a sample document ID."""
    return "test_document.md"


@pytest.fixture
def sample_query_text() -> str:
    """Return a sample query for testing."""
    return "What is the important information in section 1?"


# =============================================================================
# Corpus Fixtures
# =============================================================================


@pytest.fixture
def sample_document() -> Document:
    """Return a sample document for testing."""
    return Document(
        id=DocumentId("doc1.md"),
        content="""# Introduction to RAG Systems

Retrieval-Augmented Generation (RAG) combines retrieval with generation to provide more accurate responses.
The retrieval component fetches relevant documents from a knowledge base before generation.

## Key Benefits

RAG systems reduce hallucination by grounding responses in retrieved content.
They also allow for dynamic knowledge updates without retraining the model.

## Architecture

A typical RAG pipeline consists of three main components:
1. Document chunking and indexing
2. Semantic retrieval using embeddings
3. Response generation with context
""",
        metadata={"author": "test"},
    )


@pytest.fixture
def sample_document_2() -> Document:
    """Return a second sample document for testing."""
    return Document(
        id=DocumentId("doc2.md"),
        content="""# Vector Databases

Vector databases store and retrieve high-dimensional vectors efficiently.
They use specialized indexing algorithms like HNSW and IVF for fast similarity search.

## Popular Options

ChromaDB is an open-source embedding database designed for AI applications.
Pinecone offers a managed vector database service with serverless options.
Weaviate provides a vector search engine with built-in ML model integration.

## Use Cases

Vector databases are essential for semantic search applications.
They power recommendation systems, image search, and RAG pipelines.
""",
        metadata={"author": "test"},
    )


@pytest.fixture
def sample_corpus(sample_document: Document, sample_document_2: Document) -> Corpus:
    """Return a sample corpus with 2 documents for testing."""
    return Corpus(
        documents=[sample_document, sample_document_2],
        metadata={"name": "test_corpus", "version": "1.0"},
    )


@pytest.fixture
def sample_chunker() -> RecursiveCharacterChunker:
    """Return a RecursiveCharacterChunker with small chunk size for testing."""
    return RecursiveCharacterChunker(
        chunk_size=200,
        chunk_overlap=50,
    )


# =============================================================================
# Mock Fixtures
# =============================================================================


class MockEmbedder:
    """Mock embedder for testing without external API calls."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.call_count = 0
        self._name = "mock-embedder"

    @property
    def name(self) -> str:
        """Return the embedder name."""
        return self._name

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic mock embeddings based on text hash."""
        self.call_count += 1
        embeddings = []
        for text in texts:
            # Create deterministic embedding based on text hash
            hash_val = hash(text)
            embedding = [(hash_val + i) % 100 / 100.0 for i in range(self.dimension)]
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        return self.embed([query])[0]


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    """Return a mock embedder instance."""
    return MockEmbedder()


class MockVectorStore:
    """Mock vector store for testing without actual vector DB."""

    def __init__(self) -> None:
        self._chunks: list[PositionAwareChunk] = []
        self._embeddings: list[list[float]] = []
        self._name = "mock-vector-store"

    @property
    def name(self) -> str:
        """Return the vector store name."""
        return self._name

    def add(
        self,
        chunks: list[PositionAwareChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add chunks and embeddings to the store."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings must have same length: {len(chunks)} vs {len(embeddings)}"
            )
        self._chunks.extend(chunks)
        self._embeddings.extend(embeddings)

    def search(
        self,
        _query_embedding: list[float],
        k: int = 5,
    ) -> list[PositionAwareChunk]:
        """Return the first k chunks (simple mock behavior)."""
        return self._chunks[:k]

    def clear(self) -> None:
        """Clear all stored data."""
        self._chunks = []
        self._embeddings = []


@pytest.fixture
def mock_vector_store() -> MockVectorStore:
    """Return a mock vector store instance."""
    return MockVectorStore()


class MockLLMClient:
    """Mock LLM client for testing synthetic data generation without API calls."""

    def __init__(self, responses: list[str] | None = None) -> None:
        """Initialize with optional canned responses.

        Args:
            responses: List of responses to return in order. If None, returns default response.
        """
        self._responses = responses or []
        self._response_index = 0
        self._call_history: list[dict[str, Any]] = []
        self.chat = MagicMock()
        self.chat.completions.create = self._create_completion

    def _create_completion(self, **kwargs: Any) -> MagicMock:
        """Mock completion creation."""
        self._call_history.append(kwargs)

        # Return next canned response or default
        if self._response_index < len(self._responses):
            content = self._responses[self._response_index]
            self._response_index += 1
        else:
            content = "[]"

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = content
        return response

    def reset(self) -> None:
        """Reset call history and response index."""
        self._response_index = 0
        self._call_history = []

    @property
    def call_history(self) -> list[dict[str, Any]]:
        """Return the history of calls made."""
        return self._call_history


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    """Return a mock LLM client instance."""
    return MockLLMClient()


# =============================================================================
# Sample Ground Truth Fixtures
# =============================================================================


@pytest.fixture
def sample_character_spans(sample_document: Document) -> list[CharacterSpan]:
    """Return sample character spans from the sample document."""
    return [
        CharacterSpan(
            doc_id=sample_document.id,
            start=0,
            end=30,
            text=sample_document.content[0:30],
        ),
        CharacterSpan(
            doc_id=sample_document.id,
            start=100,
            end=150,
            text=sample_document.content[100:150],
        ),
    ]


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================


@pytest.fixture
def temp_corpus_dir(tmp_path: Any, sample_document_content: str) -> Any:
    """Create a temporary directory with sample documents."""
    # Create a few sample markdown files
    for i in range(3):
        doc_path = tmp_path / f"document_{i}.md"
        doc_path.write_text(f"# Document {i}\n\n{sample_document_content}")

    return tmp_path


# =============================================================================
# Marker-based Fixtures
# =============================================================================


@pytest.fixture
def requires_openai() -> None:
    """Skip test if OpenAI is not configured."""
    try:
        import openai  # noqa: F401
    except ImportError:
        pytest.skip("OpenAI not installed")


@pytest.fixture
def requires_cohere() -> None:
    """Skip test if Cohere is not configured."""
    try:
        import cohere  # noqa: F401
    except ImportError:
        pytest.skip("Cohere not installed")


@pytest.fixture
def requires_chromadb() -> None:
    """Skip test if ChromaDB is not configured."""
    try:
        import chromadb  # noqa: F401
    except ImportError:
        pytest.skip("ChromaDB not installed")


# =============================================================================
# Additional Corpus Fixtures
# =============================================================================


@pytest.fixture
def sample_document_3() -> Document:
    """Return a third sample document for testing."""
    return Document(
        id=DocumentId("doc3.md"),
        content="""# Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables systems to learn from data.
Supervised learning uses labeled data to train models.

## Neural Networks

Neural networks are composed of layers of interconnected nodes.
Each node applies a mathematical transformation to its inputs.

## Deep Learning

Deep learning uses neural networks with many layers.
It has achieved breakthrough results in image recognition and natural language processing.
""",
        metadata={"author": "test", "topic": "ml"},
    )


@pytest.fixture
def sample_corpus_three_docs(
    sample_document: Document,
    sample_document_2: Document,
    sample_document_3: Document,
) -> Corpus:
    """Return a sample corpus with 3 documents for testing."""
    return Corpus(
        documents=[sample_document, sample_document_2, sample_document_3],
        metadata={"name": "test_corpus_three", "version": "1.0"},
    )


@pytest.fixture
def small_chunker() -> RecursiveCharacterChunker:
    """Return a RecursiveCharacterChunker with very small chunk size for testing edge cases."""
    return RecursiveCharacterChunker(
        chunk_size=50,
        chunk_overlap=10,
    )


@pytest.fixture
def large_chunker() -> RecursiveCharacterChunker:
    """Return a RecursiveCharacterChunker with large chunk size for testing edge cases."""
    return RecursiveCharacterChunker(
        chunk_size=1000,
        chunk_overlap=200,
    )


# =============================================================================
# Ground Truth Fixtures
# =============================================================================


@pytest.fixture
def sample_chunk_level_ground_truth(sample_corpus: Corpus) -> list[Any]:
    """Return sample chunk-level ground truth for testing."""
    from rag_evaluation_system.types import (
        ChunkId,
        ChunkLevelGroundTruth,
        Query,
        QueryId,
        QueryText,
    )

    chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=50)
    chunk_ids: list[ChunkId] = []

    from rag_evaluation_system.utils.hashing import generate_chunk_id

    for doc in sample_corpus.documents:
        for chunk_text in chunker.chunk(doc.content):
            chunk_ids.append(generate_chunk_id(chunk_text))

    return [
        ChunkLevelGroundTruth(
            query=Query(
                id=QueryId("query_001"),
                text=QueryText("What is RAG?"),
                metadata={},
            ),
            relevant_chunk_ids=chunk_ids[:2] if len(chunk_ids) >= 2 else chunk_ids,
        ),
        ChunkLevelGroundTruth(
            query=Query(
                id=QueryId("query_002"),
                text=QueryText("How do vector databases work?"),
                metadata={},
            ),
            relevant_chunk_ids=chunk_ids[2:4] if len(chunk_ids) >= 4 else chunk_ids[-2:],
        ),
    ]


@pytest.fixture
def sample_token_level_ground_truth(sample_corpus: Corpus) -> list[Any]:
    """Return sample token-level ground truth for testing."""
    from rag_evaluation_system.types import (
        CharacterSpan,
        Query,
        QueryId,
        QueryText,
        TokenLevelGroundTruth,
    )

    doc1 = sample_corpus.documents[0]
    doc2 = sample_corpus.documents[1]

    return [
        TokenLevelGroundTruth(
            query=Query(
                id=QueryId("query_001"),
                text=QueryText("What is RAG?"),
                metadata={},
            ),
            relevant_spans=[
                CharacterSpan(
                    doc_id=doc1.id,
                    start=31,
                    end=105,
                    text=doc1.content[31:105],
                ),
            ],
        ),
        TokenLevelGroundTruth(
            query=Query(
                id=QueryId("query_002"),
                text=QueryText("What are vector databases?"),
                metadata={},
            ),
            relevant_spans=[
                CharacterSpan(
                    doc_id=doc2.id,
                    start=20,
                    end=100,
                    text=doc2.content[20:100],
                ),
            ],
        ),
    ]


# =============================================================================
# Mock Reranker Fixture
# =============================================================================


class MockReranker:
    """Mock reranker for testing without external API calls."""

    def __init__(self, top_k: int = 5) -> None:
        self._top_k = top_k
        self._name = "mock-reranker"
        self.call_count = 0

    @property
    def name(self) -> str:
        """Return the reranker name."""
        return self._name

    def rerank(
        self,
        _query: str,
        chunks: list[PositionAwareChunk],
        top_k: int | None = None,
    ) -> list[PositionAwareChunk]:
        """Mock reranking - just returns chunks in reverse order."""
        self.call_count += 1
        k = top_k or self._top_k
        # Simple mock: reverse the order
        return list(reversed(chunks[:k]))


@pytest.fixture
def mock_reranker() -> MockReranker:
    """Return a mock reranker instance."""
    return MockReranker()
