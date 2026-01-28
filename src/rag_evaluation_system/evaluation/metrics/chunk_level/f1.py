"""Chunk-level F1 metric implementation."""

from rag_evaluation_system.evaluation.metrics.base import ChunkLevelMetric
from rag_evaluation_system.evaluation.metrics.chunk_level.precision import (
    ChunkPrecision,
)
from rag_evaluation_system.evaluation.metrics.chunk_level.recall import ChunkRecall
from rag_evaluation_system.types import ChunkId


class ChunkF1(ChunkLevelMetric):
    """Chunk-level F1 metric.

    Computes the harmonic mean of precision and recall, providing a single
    score that balances both metrics. F1 is useful when you want to find
    a balance between precision and recall.

    Formula:
        F1 = 2 * (precision * recall) / (precision + recall)

    Example:
        If precision = 0.8 and recall = 0.6:
        F1 = 2 * (0.8 * 0.6) / (0.8 + 0.6) = 0.96 / 1.4 = 0.686
    """

    def __init__(self) -> None:
        """Initialize the F1 metric with precision and recall calculators."""
        self._precision = ChunkPrecision()
        self._recall = ChunkRecall()

    @property
    def name(self) -> str:
        """Return the metric name."""
        return "chunk_f1"

    def calculate(
        self,
        retrieved_chunk_ids: list[ChunkId],
        ground_truth_chunk_ids: list[ChunkId],
    ) -> float:
        """Calculate chunk F1 score.

        Args:
            retrieved_chunk_ids: Chunk IDs returned by the retrieval system.
            ground_truth_chunk_ids: Chunk IDs that are relevant (ground truth).

        Returns:
            F1 score in range [0.0, 1.0].
            Returns 0.0 if both precision and recall are 0.
        """
        precision = self._precision.calculate(retrieved_chunk_ids, ground_truth_chunk_ids)
        recall = self._recall.calculate(retrieved_chunk_ids, ground_truth_chunk_ids)

        if precision + recall == 0.0:
            return 0.0

        return 2 * precision * recall / (precision + recall)


__all__ = ["ChunkF1"]
