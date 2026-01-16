"""Integration tests for token-level evaluation workflow.

These tests verify the complete token-level evaluation pipeline works correctly
with mocked external dependencies (LangSmith, vector stores, embedders).
"""

from unittest.mock import MagicMock, patch

import pytest

from rag_evaluation_system.chunkers import ChunkerPositionAdapter, RecursiveCharacterChunker
from rag_evaluation_system.chunkers.base import Chunker, PositionAwareChunker
from rag_evaluation_system.evaluation.token_level import TokenLevelEvaluation
from rag_evaluation_system.types import (
    CharacterSpan,
    Corpus,
    Document,
    DocumentId,
    PositionAwareChunk,
    Query,
    QueryId,
    QueryText,
    TokenLevelGroundTruth,
)
from tests.conftest import MockEmbedder, MockVectorStore


class TestTokenLevelEvaluationEnsurePositionAware:
    """Tests for TokenLevelEvaluation._ensure_position_aware method."""

    def test_ensure_position_aware_returns_position_aware_chunker_unchanged(
        self,
        sample_corpus: Corpus,
    ) -> None:
        """Test that PositionAwareChunker is returned as-is."""
        evaluation = TokenLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        # RecursiveCharacterChunker implements PositionAwareChunker
        position_aware_chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=50)

        result = evaluation._ensure_position_aware(position_aware_chunker)

        # Should return the same instance
        assert result is position_aware_chunker

    def test_ensure_position_aware_wraps_simple_chunker(
        self,
        sample_corpus: Corpus,
    ) -> None:
        """Test that a simple Chunker is wrapped with ChunkerPositionAdapter."""
        evaluation = TokenLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        # Create a simple chunker (we'll use a mock that only implements Chunker)
        class SimpleChunker(Chunker):
            @property
            def name(self) -> str:
                return "simple-test-chunker"

            def chunk(self, text: str) -> list[str]:
                # Simple paragraph splitting
                return [p.strip() for p in text.split("\n\n") if p.strip()]

        simple_chunker = SimpleChunker()

        result = evaluation._ensure_position_aware(simple_chunker)

        # Should return a ChunkerPositionAdapter
        assert isinstance(result, ChunkerPositionAdapter)
        assert isinstance(result, PositionAwareChunker)
        assert "adapted" in result.name


class TestTokenLevelEvaluationChunkCorpus:
    """Tests for TokenLevelEvaluation._chunk_corpus method."""

    def test_chunk_corpus_produces_position_aware_chunks(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
    ) -> None:
        """Test that _chunk_corpus returns PositionAwareChunk objects."""
        evaluation = TokenLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        chunks = evaluation._chunk_corpus(sample_chunker)

        assert len(chunks) > 0
        assert all(isinstance(c, PositionAwareChunk) for c in chunks)

    def test_chunk_corpus_has_correct_positions(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
    ) -> None:
        """Test that chunks have correct position information."""
        evaluation = TokenLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        chunks = evaluation._chunk_corpus(sample_chunker)

        # Build a map of doc_id to document for validation
        doc_map = {doc.id: doc for doc in sample_corpus.documents}

        for chunk in chunks:
            assert chunk.doc_id in doc_map
            doc = doc_map[chunk.doc_id]

            # Verify the position is correct
            assert chunk.start >= 0
            assert chunk.end <= len(doc.content)
            assert chunk.content == doc.content[chunk.start : chunk.end]

    def test_chunk_corpus_uses_pa_chunk_prefix(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
    ) -> None:
        """Test that position-aware chunks use pa_chunk_ prefix."""
        evaluation = TokenLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        chunks = evaluation._chunk_corpus(sample_chunker)

        for chunk in chunks:
            assert chunk.id.startswith("pa_chunk_")

    def test_chunk_corpus_processes_all_documents(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
    ) -> None:
        """Test that all documents in corpus are chunked."""
        evaluation = TokenLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        chunks = evaluation._chunk_corpus(sample_chunker)

        # Get unique doc_ids from chunks
        doc_ids_in_chunks = {chunk.doc_id for chunk in chunks}

        # Should have chunks from all documents
        expected_doc_ids = {doc.id for doc in sample_corpus.documents}
        assert doc_ids_in_chunks == expected_doc_ids


class TestTokenLevelWorkflowIntegration:
    """Integration tests for complete token-level workflow."""

    @pytest.fixture
    def mock_ground_truth(self, sample_corpus: Corpus) -> list[TokenLevelGroundTruth]:
        """Create mock ground truth data with character spans."""
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

    def test_complete_workflow_with_mocks(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_embedder: MockEmbedder,
        mock_vector_store: MockVectorStore,
        mock_ground_truth: list[TokenLevelGroundTruth],
    ) -> None:
        """Test complete token-level workflow with mocked dependencies."""
        evaluation = TokenLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        # Mock the _load_ground_truth method
        with patch.object(evaluation, "_load_ground_truth", return_value=mock_ground_truth):
            result = evaluation.run(
                chunker=sample_chunker,
                embedder=mock_embedder,  # type: ignore[arg-type]
                k=3,
                vector_store=mock_vector_store,  # type: ignore[arg-type]
            )

        # Verify result structure
        assert result.metrics is not None
        assert "span_recall" in result.metrics
        assert "span_precision" in result.metrics
        assert "span_iou" in result.metrics

        # Metrics should be between 0 and 1
        for metric_name, score in result.metrics.items():
            assert 0.0 <= score <= 1.0, f"{metric_name} should be between 0 and 1"

    def test_workflow_converts_chunks_to_spans(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_embedder: MockEmbedder,
        mock_ground_truth: list[TokenLevelGroundTruth],
    ) -> None:
        """Test that workflow converts retrieved chunks to spans for comparison."""
        # Create a mock vector store that returns specific chunks
        doc = sample_corpus.documents[0]
        mock_chunk = PositionAwareChunk(
            id="pa_chunk_test123",
            content=doc.content[31:105],
            doc_id=doc.id,
            start=31,
            end=105,
        )

        mock_store = MockVectorStore()

        evaluation = TokenLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        # Patch _chunk_corpus to return our specific chunk
        with (
            patch.object(evaluation, "_chunk_corpus", return_value=[mock_chunk]),
            patch.object(evaluation, "_load_ground_truth", return_value=mock_ground_truth),
        ):
            # Manually add chunk to vector store
            mock_store._chunks = [mock_chunk]

            result = evaluation.run(
                chunker=sample_chunker,
                embedder=mock_embedder,  # type: ignore[arg-type]
                k=3,
                vector_store=mock_store,  # type: ignore[arg-type]
            )

        # Should have computed metrics
        assert result.metrics is not None

    def test_workflow_with_empty_ground_truth(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_embedder: MockEmbedder,
        mock_vector_store: MockVectorStore,
    ) -> None:
        """Test workflow handles empty ground truth gracefully."""
        evaluation = TokenLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        # Empty ground truth
        with patch.object(evaluation, "_load_ground_truth", return_value=[]):
            result = evaluation.run(
                chunker=sample_chunker,
                embedder=mock_embedder,  # type: ignore[arg-type]
                k=3,
                vector_store=mock_vector_store,  # type: ignore[arg-type]
            )

        # Should return 0.0 for all metrics
        assert result.metrics["span_recall"] == 0.0
        assert result.metrics["span_precision"] == 0.0
        assert result.metrics["span_iou"] == 0.0

    def test_workflow_with_simple_chunker(
        self,
        sample_corpus: Corpus,
        mock_embedder: MockEmbedder,
        mock_vector_store: MockVectorStore,
        mock_ground_truth: list[TokenLevelGroundTruth],
    ) -> None:
        """Test that workflow wraps simple Chunker with adapter."""

        class SimpleChunker(Chunker):
            @property
            def name(self) -> str:
                return "simple"

            def chunk(self, text: str) -> list[str]:
                # Split on double newlines
                return [p.strip() for p in text.split("\n\n") if p.strip()]

        simple_chunker = SimpleChunker()

        evaluation = TokenLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        with patch.object(evaluation, "_load_ground_truth", return_value=mock_ground_truth):
            result = evaluation.run(
                chunker=simple_chunker,
                embedder=mock_embedder,  # type: ignore[arg-type]
                k=3,
                vector_store=mock_vector_store,  # type: ignore[arg-type]
            )

        # Should work without errors
        assert result.metrics is not None


class TestTokenLevelChunkToSpanConversion:
    """Tests for chunk to span conversion in token-level evaluation."""

    def test_position_aware_chunk_to_span(
        self,
        sample_document: Document,
    ) -> None:
        """Test that PositionAwareChunk.to_span() works correctly."""
        chunk = PositionAwareChunk(
            id="pa_chunk_test123",
            content="Test content",
            doc_id=sample_document.id,
            start=0,
            end=12,
        )

        span = chunk.to_span()

        assert span.doc_id == chunk.doc_id
        assert span.start == chunk.start
        assert span.end == chunk.end
        assert span.text == chunk.content


@pytest.mark.integration
class TestTokenLevelLoadGroundTruth:
    """Tests for _load_ground_truth method (requires mocking LangSmith)."""

    def test_load_ground_truth_with_mock_langsmith(
        self,
        sample_corpus: Corpus,
    ) -> None:
        """Test that _load_ground_truth correctly loads data from LangSmith."""
        evaluation = TokenLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        # Create mock LangSmith client and data
        mock_example = MagicMock()
        mock_example.id = "example-001"
        mock_example.inputs = {"query": "What is RAG?"}
        mock_example.outputs = {
            "relevant_spans": [
                {
                    "doc_id": "doc1.md",
                    "start": 31,
                    "end": 105,
                    "text": "Test span text that is exactly 74 characters long test text aaaaaaaaaaaaaX",
                },
            ]
        }
        mock_example.metadata = {"source": "synthetic"}

        mock_dataset = MagicMock()
        mock_dataset.id = "dataset-001"

        mock_client = MagicMock()
        mock_client.read_dataset.return_value = mock_dataset
        mock_client.list_examples.return_value = [mock_example]

        # Patch the Client class in langsmith module (where it's imported from)
        with patch("langsmith.Client", return_value=mock_client):
            ground_truth = evaluation._load_ground_truth()

        assert len(ground_truth) == 1
        gt = ground_truth[0]

        assert gt.query.text == "What is RAG?"
        assert len(gt.relevant_spans) == 1

        span = gt.relevant_spans[0]
        assert span.doc_id == DocumentId("doc1.md")
        assert span.start == 31
        assert span.end == 105

    def test_load_ground_truth_handles_multiple_spans(
        self,
        sample_corpus: Corpus,
    ) -> None:
        """Test that _load_ground_truth handles examples with multiple spans."""
        evaluation = TokenLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        mock_example = MagicMock()
        mock_example.id = "example-001"
        mock_example.inputs = {"query": "Complex question"}
        mock_example.outputs = {
            "relevant_spans": [
                {"doc_id": "doc1.md", "start": 0, "end": 10, "text": "0123456789"},
                {"doc_id": "doc1.md", "start": 50, "end": 70, "text": "01234567890123456789"},
                {
                    "doc_id": "doc2.md",
                    "start": 100,
                    "end": 125,
                    "text": "0123456789012345678901234",
                },
            ]
        }
        mock_example.metadata = {}

        mock_dataset = MagicMock()
        mock_dataset.id = "dataset-001"

        mock_client = MagicMock()
        mock_client.read_dataset.return_value = mock_dataset
        mock_client.list_examples.return_value = [mock_example]

        # Patch the Client class in langsmith module (where it's imported from)
        with patch("langsmith.Client", return_value=mock_client):
            ground_truth = evaluation._load_ground_truth()

        assert len(ground_truth) == 1
        assert len(ground_truth[0].relevant_spans) == 3


class TestTokenLevelWrappedChunker:
    """Tests for ChunkerPositionAdapter integration."""

    def test_adapted_chunker_finds_positions(
        self,
        sample_document: Document,
    ) -> None:
        """Test that adapted chunker correctly finds chunk positions."""

        class ParagraphChunker(Chunker):
            @property
            def name(self) -> str:
                return "paragraph"

            def chunk(self, text: str) -> list[str]:
                return [p.strip() for p in text.split("\n\n") if p.strip()]

        base_chunker = ParagraphChunker()
        adapter = ChunkerPositionAdapter(base_chunker)

        chunks = adapter.chunk_with_positions(sample_document)

        assert len(chunks) > 0

        for chunk in chunks:
            # Verify position is correct
            assert sample_document.content[chunk.start : chunk.end] == chunk.content
            assert chunk.doc_id == sample_document.id

    def test_adapted_chunker_skipped_count(
        self,
    ) -> None:
        """Test that adapted chunker tracks skipped chunks."""

        class ModifyingChunker(Chunker):
            """A chunker that modifies text (bad practice, but tests adapter)."""

            @property
            def name(self) -> str:
                return "modifying"

            def chunk(self, text: str) -> list[str]:
                # Return modified text that won't be found
                return ["MODIFIED: " + text[:50]]

        doc = Document(
            id=DocumentId("test.md"),
            content="Original content that will be modified by the chunker.",
        )

        base_chunker = ModifyingChunker()
        adapter = ChunkerPositionAdapter(base_chunker)

        chunks = adapter.chunk_with_positions(doc)

        # Chunk was modified, so it won't be found
        assert len(chunks) == 0
        assert adapter.skipped_chunks == 1


class TestTokenLevelWorkflowWithCustomMetrics:
    """Tests for workflow with custom metrics."""

    def test_workflow_with_single_metric(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_embedder: MockEmbedder,
        mock_vector_store: MockVectorStore,
    ) -> None:
        """Test workflow with only recall metric."""
        from rag_evaluation_system.evaluation.metrics import SpanRecall

        doc = sample_corpus.documents[0]

        mock_ground_truth = [
            TokenLevelGroundTruth(
                query=Query(
                    id=QueryId("query_001"),
                    text=QueryText("What is RAG?"),
                    metadata={},
                ),
                relevant_spans=[
                    CharacterSpan(
                        doc_id=doc.id,
                        start=31,
                        end=105,
                        text=doc.content[31:105],
                    ),
                ],
            ),
        ]

        evaluation = TokenLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        with patch.object(evaluation, "_load_ground_truth", return_value=mock_ground_truth):
            result = evaluation.run(
                chunker=sample_chunker,
                embedder=mock_embedder,  # type: ignore[arg-type]
                k=3,
                vector_store=mock_vector_store,  # type: ignore[arg-type]
                metrics=[SpanRecall()],
            )

        # Should only have recall metric
        assert "span_recall" in result.metrics
        # Should not have precision or IoU
        assert "span_precision" not in result.metrics
        assert "span_iou" not in result.metrics


class TestTokenLevelWorkflowRawResults:
    """Tests for workflow raw results."""

    def test_workflow_returns_raw_results(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_embedder: MockEmbedder,
        mock_vector_store: MockVectorStore,
    ) -> None:
        """Test that workflow returns raw per-query results."""
        doc1 = sample_corpus.documents[0]
        doc2 = sample_corpus.documents[1]

        mock_ground_truth = [
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

        evaluation = TokenLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        with patch.object(evaluation, "_load_ground_truth", return_value=mock_ground_truth):
            result = evaluation.run(
                chunker=sample_chunker,
                embedder=mock_embedder,  # type: ignore[arg-type]
                k=3,
                vector_store=mock_vector_store,  # type: ignore[arg-type]
            )

        # Should have raw results
        assert result.raw_results is not None
        assert "span_recall" in result.raw_results
        assert "span_precision" in result.raw_results
        assert "span_iou" in result.raw_results

        # Each should have one score per query (2 queries)
        assert len(result.raw_results["span_recall"]) == 2
        assert len(result.raw_results["span_precision"]) == 2
        assert len(result.raw_results["span_iou"]) == 2


class TestTokenLevelChunkWithMultipleDocuments:
    """Tests for chunking multiple documents in token-level evaluation."""

    def test_chunk_corpus_assigns_correct_doc_ids(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
    ) -> None:
        """Test that _chunk_corpus assigns correct document IDs to chunks."""
        evaluation = TokenLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        chunks = evaluation._chunk_corpus(sample_chunker)

        # Collect unique doc_ids from chunks
        doc_ids_in_chunks = {chunk.doc_id for chunk in chunks}

        # Should have chunks from both documents
        expected_doc_ids = {doc.id for doc in sample_corpus.documents}
        assert doc_ids_in_chunks == expected_doc_ids

    def test_chunk_positions_are_within_document_bounds(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
    ) -> None:
        """Test that chunk positions are valid within document bounds."""
        evaluation = TokenLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        chunks = evaluation._chunk_corpus(sample_chunker)

        # Build a map of doc_id to document
        doc_map = {doc.id: doc for doc in sample_corpus.documents}

        for chunk in chunks:
            doc = doc_map[chunk.doc_id]
            assert chunk.start >= 0
            assert chunk.end <= len(doc.content)
            assert chunk.start < chunk.end
