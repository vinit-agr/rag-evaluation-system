"""Integration tests for chunk-level evaluation workflow.

These tests verify the complete chunk-level evaluation pipeline works correctly
with mocked external dependencies (LangSmith, vector stores, embedders).
"""

from unittest.mock import MagicMock, patch

import pytest

from rag_evaluation_system.chunkers import RecursiveCharacterChunker
from rag_evaluation_system.evaluation.chunk_level import ChunkLevelEvaluation
from rag_evaluation_system.types import (
    ChunkId,
    ChunkLevelGroundTruth,
    Corpus,
    DocumentId,
    PositionAwareChunk,
    Query,
    QueryId,
    QueryText,
)
from rag_evaluation_system.utils.hashing import generate_chunk_id, generate_pa_chunk_id
from tests.conftest import MockEmbedder, MockVectorStore


class TestChunkLevelEvaluationChunkCorpus:
    """Tests for ChunkLevelEvaluation._chunk_corpus method."""

    def test_chunk_corpus_produces_chunks_and_ids(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
    ) -> None:
        """Test that _chunk_corpus returns list of chunk texts and ChunkIds."""
        evaluation = ChunkLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        chunks, chunk_ids = evaluation._chunk_corpus(sample_chunker)

        # Should have chunks from both documents
        assert len(chunks) > 0
        assert len(chunks) == len(chunk_ids)

        # All chunk IDs should have correct prefix
        for chunk_id in chunk_ids:
            assert chunk_id.startswith("chunk_")

    def test_chunk_corpus_ids_are_content_based(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
    ) -> None:
        """Test that chunk IDs are deterministic based on content."""
        evaluation = ChunkLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        # Run chunking twice
        chunks1, ids1 = evaluation._chunk_corpus(sample_chunker)
        chunks2, ids2 = evaluation._chunk_corpus(sample_chunker)

        # Should produce same results
        assert chunks1 == chunks2
        assert ids1 == ids2

    def test_chunk_corpus_matches_generate_chunk_id(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
    ) -> None:
        """Test that chunk IDs match generate_chunk_id output."""
        evaluation = ChunkLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        chunks, chunk_ids = evaluation._chunk_corpus(sample_chunker)

        for chunk_text, chunk_id in zip(chunks, chunk_ids, strict=True):
            expected_id = generate_chunk_id(chunk_text)
            assert chunk_id == expected_id


class TestChunkLevelEvaluationToPositionAware:
    """Tests for ChunkLevelEvaluation._to_position_aware method."""

    def test_to_position_aware_creates_position_aware_chunks(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
    ) -> None:
        """Test that _to_position_aware creates PositionAwareChunk objects."""
        evaluation = ChunkLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        chunks, chunk_ids = evaluation._chunk_corpus(sample_chunker)
        pa_chunks = evaluation._to_position_aware(chunks, chunk_ids)

        assert len(pa_chunks) == len(chunks)
        assert all(isinstance(c, PositionAwareChunk) for c in pa_chunks)

    def test_to_position_aware_uses_pa_chunk_prefix(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
    ) -> None:
        """Test that position-aware chunks use pa_chunk_ prefix."""
        evaluation = ChunkLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        chunks, chunk_ids = evaluation._chunk_corpus(sample_chunker)
        pa_chunks = evaluation._to_position_aware(chunks, chunk_ids)

        for pa_chunk in pa_chunks:
            assert pa_chunk.id.startswith("pa_chunk_")

    def test_to_position_aware_preserves_content(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
    ) -> None:
        """Test that position-aware chunks have correct content."""
        evaluation = ChunkLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        chunks, chunk_ids = evaluation._chunk_corpus(sample_chunker)
        pa_chunks = evaluation._to_position_aware(chunks, chunk_ids)

        for chunk_text, pa_chunk in zip(chunks, pa_chunks, strict=True):
            assert pa_chunk.content == chunk_text

    def test_to_position_aware_stores_original_chunk_id(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
    ) -> None:
        """Test that original chunk ID is stored in metadata."""
        evaluation = ChunkLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        chunks, chunk_ids = evaluation._chunk_corpus(sample_chunker)
        pa_chunks = evaluation._to_position_aware(chunks, chunk_ids)

        for chunk_id, pa_chunk in zip(chunk_ids, pa_chunks, strict=True):
            assert pa_chunk.metadata.get("original_chunk_id") == chunk_id


class TestChunkLevelWorkflowIntegration:
    """Integration tests for complete chunk-level workflow."""

    @pytest.fixture
    def mock_ground_truth(self, sample_corpus: Corpus) -> list[ChunkLevelGroundTruth]:
        """Create mock ground truth data."""
        # Get actual chunk IDs from the corpus
        chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=50)
        chunk_ids: list[ChunkId] = []

        for doc in sample_corpus.documents:
            for chunk_text in chunker.chunk(doc.content):
                chunk_ids.append(generate_chunk_id(chunk_text))

        # Create ground truth with first few chunk IDs
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

    def test_complete_workflow_with_mocks(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_embedder: MockEmbedder,
        mock_vector_store: MockVectorStore,
        mock_ground_truth: list[ChunkLevelGroundTruth],
    ) -> None:
        """Test complete chunk-level workflow with mocked dependencies."""
        evaluation = ChunkLevelEvaluation(
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
        assert "chunk_recall" in result.metrics
        assert "chunk_precision" in result.metrics
        assert "chunk_f1" in result.metrics

        # Metrics should be between 0 and 1
        for metric_name, score in result.metrics.items():
            assert 0.0 <= score <= 1.0, f"{metric_name} should be between 0 and 1"

    def test_workflow_chunks_all_documents(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_embedder: MockEmbedder,
        mock_vector_store: MockVectorStore,
        mock_ground_truth: list[ChunkLevelGroundTruth],
    ) -> None:
        """Test that workflow chunks all documents in corpus."""
        evaluation = ChunkLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        with patch.object(evaluation, "_load_ground_truth", return_value=mock_ground_truth):
            evaluation.run(
                chunker=sample_chunker,
                embedder=mock_embedder,  # type: ignore[arg-type]
                k=3,
                vector_store=mock_vector_store,  # type: ignore[arg-type]
            )

        # Vector store should have chunks from all documents
        assert len(mock_vector_store._chunks) > 0

    def test_workflow_embeds_all_chunks(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_embedder: MockEmbedder,
        mock_vector_store: MockVectorStore,
        mock_ground_truth: list[ChunkLevelGroundTruth],
    ) -> None:
        """Test that workflow embeds all chunks."""
        evaluation = ChunkLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        with patch.object(evaluation, "_load_ground_truth", return_value=mock_ground_truth):
            evaluation.run(
                chunker=sample_chunker,
                embedder=mock_embedder,  # type: ignore[arg-type]
                k=3,
                vector_store=mock_vector_store,  # type: ignore[arg-type]
            )

        # Embedder should have been called
        assert mock_embedder.call_count > 0

    def test_workflow_with_empty_ground_truth(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_embedder: MockEmbedder,
        mock_vector_store: MockVectorStore,
    ) -> None:
        """Test workflow handles empty ground truth gracefully."""
        evaluation = ChunkLevelEvaluation(
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
        assert result.metrics["chunk_recall"] == 0.0
        assert result.metrics["chunk_precision"] == 0.0
        assert result.metrics["chunk_f1"] == 0.0


class TestChunkLevelConvertToChunkIds:
    """Tests for _convert_to_chunk_ids method."""

    def test_convert_to_chunk_ids_replaces_prefix(
        self,
        sample_corpus: Corpus,
    ) -> None:
        """Test that _convert_to_chunk_ids correctly replaces pa_chunk_ with chunk_."""
        evaluation = ChunkLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        # Create position-aware chunks with pa_chunk_ prefix
        pa_chunks = [
            PositionAwareChunk(
                id=generate_pa_chunk_id("test content 1"),
                content="test content 1",
                doc_id=DocumentId("doc1"),
                start=0,
                end=14,
            ),
            PositionAwareChunk(
                id=generate_pa_chunk_id("test content 2"),
                content="test content 2",
                doc_id=DocumentId("doc1"),
                start=20,
                end=34,
            ),
        ]

        chunk_ids = evaluation._convert_to_chunk_ids(pa_chunks)

        # All should now have chunk_ prefix
        for chunk_id in chunk_ids:
            assert chunk_id.startswith("chunk_")
            assert not chunk_id.startswith("pa_chunk_")

    def test_convert_to_chunk_ids_preserves_hash(
        self,
        sample_corpus: Corpus,
    ) -> None:
        """Test that conversion preserves the content hash portion of the ID."""
        evaluation = ChunkLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        content = "test content"
        pa_chunk_id = generate_pa_chunk_id(content)
        chunk_id = generate_chunk_id(content)

        pa_chunks = [
            PositionAwareChunk(
                id=pa_chunk_id,
                content=content,
                doc_id=DocumentId("doc1"),
                start=0,
                end=len(content),
            ),
        ]

        converted_ids = evaluation._convert_to_chunk_ids(pa_chunks)

        # The hash portion should match
        assert converted_ids[0] == chunk_id


@pytest.mark.integration
class TestChunkLevelLoadGroundTruth:
    """Tests for _load_ground_truth method (requires mocking LangSmith)."""

    def test_load_ground_truth_with_mock_langsmith(
        self,
        sample_corpus: Corpus,
    ) -> None:
        """Test that _load_ground_truth correctly loads data from LangSmith."""
        evaluation = ChunkLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        # Create mock LangSmith client and data
        mock_example = MagicMock()
        mock_example.id = "example-001"
        mock_example.inputs = {"query": "What is RAG?"}
        mock_example.outputs = {"relevant_chunk_ids": ["chunk_abc123", "chunk_def456"]}
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
        assert len(gt.relevant_chunk_ids) == 2
        assert ChunkId("chunk_abc123") in gt.relevant_chunk_ids
        assert ChunkId("chunk_def456") in gt.relevant_chunk_ids

    def test_load_ground_truth_handles_multiple_examples(
        self,
        sample_corpus: Corpus,
    ) -> None:
        """Test that _load_ground_truth handles multiple examples."""
        evaluation = ChunkLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        # Create multiple mock examples
        mock_example1 = MagicMock()
        mock_example1.id = "example-001"
        mock_example1.inputs = {"query": "What is RAG?"}
        mock_example1.outputs = {"relevant_chunk_ids": ["chunk_abc123"]}
        mock_example1.metadata = {}

        mock_example2 = MagicMock()
        mock_example2.id = "example-002"
        mock_example2.inputs = {"query": "What are embeddings?"}
        mock_example2.outputs = {"relevant_chunk_ids": ["chunk_xyz789", "chunk_def456"]}
        mock_example2.metadata = {}

        mock_dataset = MagicMock()
        mock_dataset.id = "dataset-001"

        mock_client = MagicMock()
        mock_client.read_dataset.return_value = mock_dataset
        mock_client.list_examples.return_value = [mock_example1, mock_example2]

        with patch("langsmith.Client", return_value=mock_client):
            ground_truth = evaluation._load_ground_truth()

        assert len(ground_truth) == 2
        assert ground_truth[0].query.text == "What is RAG?"
        assert ground_truth[1].query.text == "What are embeddings?"

    def test_load_ground_truth_handles_empty_metadata(
        self,
        sample_corpus: Corpus,
    ) -> None:
        """Test that _load_ground_truth handles examples with None metadata."""
        evaluation = ChunkLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        mock_example = MagicMock()
        mock_example.id = "example-001"
        mock_example.inputs = {"query": "Test query?"}
        mock_example.outputs = {"relevant_chunk_ids": ["chunk_123"]}
        mock_example.metadata = None  # None metadata

        mock_dataset = MagicMock()
        mock_dataset.id = "dataset-001"

        mock_client = MagicMock()
        mock_client.read_dataset.return_value = mock_dataset
        mock_client.list_examples.return_value = [mock_example]

        with patch("langsmith.Client", return_value=mock_client):
            ground_truth = evaluation._load_ground_truth()

        assert len(ground_truth) == 1
        assert ground_truth[0].query.metadata == {}


class TestChunkLevelWorkflowWithCustomMetrics:
    """Tests for workflow with custom metrics."""

    def test_workflow_with_single_metric(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_embedder: MockEmbedder,
        mock_vector_store: MockVectorStore,
    ) -> None:
        """Test workflow with only recall metric."""
        from rag_evaluation_system.evaluation.metrics import ChunkRecall

        # Create simple ground truth
        chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=50)
        chunk_ids: list[ChunkId] = []
        for doc in sample_corpus.documents:
            for chunk_text in chunker.chunk(doc.content):
                chunk_ids.append(generate_chunk_id(chunk_text))

        mock_ground_truth = [
            ChunkLevelGroundTruth(
                query=Query(
                    id=QueryId("query_001"),
                    text=QueryText("What is RAG?"),
                    metadata={},
                ),
                relevant_chunk_ids=chunk_ids[:1] if chunk_ids else [],
            ),
        ]

        evaluation = ChunkLevelEvaluation(
            corpus=sample_corpus,
            langsmith_dataset_name="test-dataset",
        )

        with patch.object(evaluation, "_load_ground_truth", return_value=mock_ground_truth):
            result = evaluation.run(
                chunker=sample_chunker,
                embedder=mock_embedder,  # type: ignore[arg-type]
                k=3,
                vector_store=mock_vector_store,  # type: ignore[arg-type]
                metrics=[ChunkRecall()],
            )

        # Should only have recall metric
        assert "chunk_recall" in result.metrics
        # Should not have precision or F1
        assert "chunk_precision" not in result.metrics
        assert "chunk_f1" not in result.metrics


class TestChunkLevelWorkflowRawResults:
    """Tests for workflow raw results."""

    def test_workflow_returns_raw_results(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_embedder: MockEmbedder,
        mock_vector_store: MockVectorStore,
    ) -> None:
        """Test that workflow returns raw per-query results."""
        # Create ground truth
        chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=50)
        chunk_ids: list[ChunkId] = []
        for doc in sample_corpus.documents:
            for chunk_text in chunker.chunk(doc.content):
                chunk_ids.append(generate_chunk_id(chunk_text))

        mock_ground_truth = [
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
                    text=QueryText("What are embeddings?"),
                    metadata={},
                ),
                relevant_chunk_ids=chunk_ids[2:4] if len(chunk_ids) >= 4 else chunk_ids[-2:],
            ),
        ]

        evaluation = ChunkLevelEvaluation(
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
        assert "chunk_recall" in result.raw_results
        assert "chunk_precision" in result.raw_results
        assert "chunk_f1" in result.raw_results

        # Each should have one score per query (2 queries)
        assert len(result.raw_results["chunk_recall"]) == 2
        assert len(result.raw_results["chunk_precision"]) == 2
        assert len(result.raw_results["chunk_f1"]) == 2
