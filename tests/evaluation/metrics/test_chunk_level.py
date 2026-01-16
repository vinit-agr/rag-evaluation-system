"""Tests for chunk-level metrics: ChunkRecall, ChunkPrecision, ChunkF1."""

import pytest

from rag_evaluation_system.evaluation.metrics.chunk_level import (
    ChunkF1,
    ChunkPrecision,
    ChunkRecall,
)
from rag_evaluation_system.types import ChunkId


class TestChunkRecall:
    """Tests for the ChunkRecall metric."""

    def test_chunk_recall_name(self) -> None:
        """Test that ChunkRecall has the correct name."""
        metric = ChunkRecall()
        assert metric.name == "chunk_recall"

    def test_chunk_recall_perfect_recall(self) -> None:
        """Test recall is 1.0 when all ground truth chunks are retrieved."""
        metric = ChunkRecall()
        ground_truth = [ChunkId("chunk_a"), ChunkId("chunk_b"), ChunkId("chunk_c")]
        retrieved = [
            ChunkId("chunk_a"),
            ChunkId("chunk_b"),
            ChunkId("chunk_c"),
            ChunkId("chunk_d"),
        ]

        result = metric.calculate(retrieved, ground_truth)

        assert result == 1.0

    def test_chunk_recall_partial_recall(self) -> None:
        """Test recall calculation with partial overlap."""
        metric = ChunkRecall()
        ground_truth = [ChunkId("chunk_a"), ChunkId("chunk_b"), ChunkId("chunk_c")]
        retrieved = [ChunkId("chunk_a"), ChunkId("chunk_b"), ChunkId("chunk_d")]

        result = metric.calculate(retrieved, ground_truth)

        # 2 out of 3 ground truth chunks retrieved
        assert result == pytest.approx(2 / 3)

    def test_chunk_recall_no_recall(self) -> None:
        """Test recall is 0.0 when no ground truth chunks are retrieved."""
        metric = ChunkRecall()
        ground_truth = [ChunkId("chunk_a"), ChunkId("chunk_b"), ChunkId("chunk_c")]
        retrieved = [ChunkId("chunk_x"), ChunkId("chunk_y"), ChunkId("chunk_z")]

        result = metric.calculate(retrieved, ground_truth)

        assert result == 0.0

    def test_chunk_recall_empty_ground_truth(self) -> None:
        """Test recall is 0.0 when ground truth is empty."""
        metric = ChunkRecall()
        ground_truth: list[ChunkId] = []
        retrieved = [ChunkId("chunk_a"), ChunkId("chunk_b")]

        result = metric.calculate(retrieved, ground_truth)

        assert result == 0.0

    def test_chunk_recall_empty_retrieved(self) -> None:
        """Test recall is 0.0 when retrieved is empty but ground truth is not."""
        metric = ChunkRecall()
        ground_truth = [ChunkId("chunk_a"), ChunkId("chunk_b")]
        retrieved: list[ChunkId] = []

        result = metric.calculate(retrieved, ground_truth)

        assert result == 0.0

    def test_chunk_recall_both_empty(self) -> None:
        """Test recall is 0.0 when both lists are empty."""
        metric = ChunkRecall()
        ground_truth: list[ChunkId] = []
        retrieved: list[ChunkId] = []

        result = metric.calculate(retrieved, ground_truth)

        assert result == 0.0

    def test_chunk_recall_duplicates_in_retrieved(self) -> None:
        """Test that duplicates in retrieved list are handled correctly."""
        metric = ChunkRecall()
        ground_truth = [ChunkId("chunk_a"), ChunkId("chunk_b")]
        # Same chunk retrieved multiple times
        retrieved = [ChunkId("chunk_a"), ChunkId("chunk_a"), ChunkId("chunk_a")]

        result = metric.calculate(retrieved, ground_truth)

        # Only chunk_a is relevant, so recall is 1/2
        assert result == 0.5

    def test_chunk_recall_exact_match(self) -> None:
        """Test recall is 1.0 when retrieved exactly matches ground truth."""
        metric = ChunkRecall()
        chunks = [ChunkId("chunk_a"), ChunkId("chunk_b")]

        result = metric.calculate(chunks, chunks)

        assert result == 1.0


class TestChunkPrecision:
    """Tests for the ChunkPrecision metric."""

    def test_chunk_precision_name(self) -> None:
        """Test that ChunkPrecision has the correct name."""
        metric = ChunkPrecision()
        assert metric.name == "chunk_precision"

    def test_chunk_precision_perfect_precision(self) -> None:
        """Test precision is 1.0 when all retrieved chunks are relevant."""
        metric = ChunkPrecision()
        ground_truth = [
            ChunkId("chunk_a"),
            ChunkId("chunk_b"),
            ChunkId("chunk_c"),
            ChunkId("chunk_d"),
        ]
        retrieved = [ChunkId("chunk_a"), ChunkId("chunk_b"), ChunkId("chunk_c")]

        result = metric.calculate(retrieved, ground_truth)

        assert result == 1.0

    def test_chunk_precision_partial_precision(self) -> None:
        """Test precision calculation with partial overlap."""
        metric = ChunkPrecision()
        ground_truth = [ChunkId("chunk_a"), ChunkId("chunk_b"), ChunkId("chunk_c")]
        retrieved = [ChunkId("chunk_a"), ChunkId("chunk_b"), ChunkId("chunk_d")]

        result = metric.calculate(retrieved, ground_truth)

        # 2 out of 3 retrieved chunks are relevant
        assert result == pytest.approx(2 / 3)

    def test_chunk_precision_no_precision(self) -> None:
        """Test precision is 0.0 when no retrieved chunks are relevant."""
        metric = ChunkPrecision()
        ground_truth = [ChunkId("chunk_a"), ChunkId("chunk_b"), ChunkId("chunk_c")]
        retrieved = [ChunkId("chunk_x"), ChunkId("chunk_y"), ChunkId("chunk_z")]

        result = metric.calculate(retrieved, ground_truth)

        assert result == 0.0

    def test_chunk_precision_empty_retrieved(self) -> None:
        """Test precision is 0.0 when retrieved is empty."""
        metric = ChunkPrecision()
        ground_truth = [ChunkId("chunk_a"), ChunkId("chunk_b")]
        retrieved: list[ChunkId] = []

        result = metric.calculate(retrieved, ground_truth)

        assert result == 0.0

    def test_chunk_precision_empty_ground_truth(self) -> None:
        """Test precision is 0.0 when ground truth is empty."""
        metric = ChunkPrecision()
        ground_truth: list[ChunkId] = []
        retrieved = [ChunkId("chunk_a"), ChunkId("chunk_b")]

        result = metric.calculate(retrieved, ground_truth)

        # All retrieved chunks are "irrelevant" since ground truth is empty
        assert result == 0.0

    def test_chunk_precision_both_empty(self) -> None:
        """Test precision is 0.0 when both lists are empty."""
        metric = ChunkPrecision()
        ground_truth: list[ChunkId] = []
        retrieved: list[ChunkId] = []

        result = metric.calculate(retrieved, ground_truth)

        assert result == 0.0

    def test_chunk_precision_duplicates_handled(self) -> None:
        """Test that duplicates in lists are handled via set conversion."""
        metric = ChunkPrecision()
        ground_truth = [ChunkId("chunk_a"), ChunkId("chunk_a")]
        retrieved = [ChunkId("chunk_a"), ChunkId("chunk_b"), ChunkId("chunk_b")]

        result = metric.calculate(retrieved, ground_truth)

        # After set conversion: retrieved = {a, b}, gt = {a}
        # Intersection = {a}, precision = 1/2
        assert result == 0.5


class TestChunkF1:
    """Tests for the ChunkF1 metric."""

    def test_chunk_f1_name(self) -> None:
        """Test that ChunkF1 has the correct name."""
        metric = ChunkF1()
        assert metric.name == "chunk_f1"

    def test_chunk_f1_perfect_score(self) -> None:
        """Test F1 is 1.0 when precision and recall are both 1.0."""
        metric = ChunkF1()
        chunks = [ChunkId("chunk_a"), ChunkId("chunk_b"), ChunkId("chunk_c")]

        result = metric.calculate(chunks, chunks)

        assert result == 1.0

    def test_chunk_f1_balanced_scenario(self) -> None:
        """Test F1 calculation with balanced precision and recall."""
        metric = ChunkF1()
        ground_truth = [ChunkId("chunk_a"), ChunkId("chunk_b")]
        retrieved = [ChunkId("chunk_a"), ChunkId("chunk_c")]

        result = metric.calculate(retrieved, ground_truth)

        # Precision = 1/2 (chunk_a is relevant out of 2 retrieved)
        # Recall = 1/2 (chunk_a retrieved out of 2 ground truth)
        # F1 = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5
        assert result == pytest.approx(0.5)

    def test_chunk_f1_high_precision_low_recall(self) -> None:
        """Test F1 with high precision but low recall."""
        metric = ChunkF1()
        ground_truth = [
            ChunkId("chunk_a"),
            ChunkId("chunk_b"),
            ChunkId("chunk_c"),
            ChunkId("chunk_d"),
        ]
        retrieved = [ChunkId("chunk_a")]  # Only one relevant chunk retrieved

        result = metric.calculate(retrieved, ground_truth)

        # Precision = 1.0 (1 relevant / 1 retrieved)
        # Recall = 0.25 (1 found / 4 ground truth)
        # F1 = 2 * (1.0 * 0.25) / (1.0 + 0.25) = 0.4
        assert result == pytest.approx(0.4)

    def test_chunk_f1_low_precision_high_recall(self) -> None:
        """Test F1 with low precision but high recall."""
        metric = ChunkF1()
        ground_truth = [ChunkId("chunk_a")]
        retrieved = [
            ChunkId("chunk_a"),
            ChunkId("chunk_b"),
            ChunkId("chunk_c"),
            ChunkId("chunk_d"),
        ]

        result = metric.calculate(retrieved, ground_truth)

        # Precision = 0.25 (1 relevant / 4 retrieved)
        # Recall = 1.0 (1 found / 1 ground truth)
        # F1 = 2 * (0.25 * 1.0) / (0.25 + 1.0) = 0.4
        assert result == pytest.approx(0.4)

    def test_chunk_f1_zero_precision_and_recall(self) -> None:
        """Test F1 is 0.0 when both precision and recall are 0."""
        metric = ChunkF1()
        ground_truth = [ChunkId("chunk_a"), ChunkId("chunk_b")]
        retrieved = [ChunkId("chunk_x"), ChunkId("chunk_y")]

        result = metric.calculate(retrieved, ground_truth)

        assert result == 0.0

    def test_chunk_f1_empty_retrieved(self) -> None:
        """Test F1 is 0.0 when retrieved is empty."""
        metric = ChunkF1()
        ground_truth = [ChunkId("chunk_a"), ChunkId("chunk_b")]
        retrieved: list[ChunkId] = []

        result = metric.calculate(retrieved, ground_truth)

        assert result == 0.0

    def test_chunk_f1_empty_ground_truth(self) -> None:
        """Test F1 is 0.0 when ground truth is empty."""
        metric = ChunkF1()
        ground_truth: list[ChunkId] = []
        retrieved = [ChunkId("chunk_a"), ChunkId("chunk_b")]

        result = metric.calculate(retrieved, ground_truth)

        assert result == 0.0

    def test_chunk_f1_both_empty(self) -> None:
        """Test F1 is 0.0 when both lists are empty."""
        metric = ChunkF1()
        ground_truth: list[ChunkId] = []
        retrieved: list[ChunkId] = []

        result = metric.calculate(retrieved, ground_truth)

        assert result == 0.0

    def test_chunk_f1_known_values(self) -> None:
        """Test F1 calculation with known precision and recall values."""
        metric = ChunkF1()
        # Setup: 3 ground truth, 3 retrieved, 2 in common
        ground_truth = [ChunkId("chunk_a"), ChunkId("chunk_b"), ChunkId("chunk_c")]
        retrieved = [ChunkId("chunk_a"), ChunkId("chunk_b"), ChunkId("chunk_d")]

        result = metric.calculate(retrieved, ground_truth)

        # Precision = 2/3
        # Recall = 2/3
        # F1 = 2 * (2/3 * 2/3) / (2/3 + 2/3) = 2 * (4/9) / (4/3) = (8/9) / (4/3) = 2/3
        assert result == pytest.approx(2 / 3)
