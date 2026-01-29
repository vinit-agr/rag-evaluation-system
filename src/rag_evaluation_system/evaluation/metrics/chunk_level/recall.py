"""Chunk-level recall metric implementation."""

from rag_evaluation_system.evaluation.metrics.base import ChunkLevelMetric
from rag_evaluation_system.types import ChunkId


class ChunkRecall(ChunkLevelMetric):
    """Chunk-level recall metric.

    Measures the proportion of ground truth chunks that were successfully
    retrieved. A recall of 1.0 means all relevant chunks were found.

    Formula:
        recall = |retrieved ∩ ground_truth| / |ground_truth|

    Example:
        If ground_truth = {A, B, C} and retrieved = {A, B, D}:
        intersection = {A, B}
        recall = 2 / 3 = 0.667
    """

    @property
    def name(self) -> str:
        """Return the metric name."""
        return "chunk_recall"

    def calculate(
        self,
        retrieved_chunk_ids: list[ChunkId],
        ground_truth_chunk_ids: list[ChunkId],
    ) -> float:
        """Calculate chunk recall.

        Args:
            retrieved_chunk_ids: Chunk IDs returned by the retrieval system.
            ground_truth_chunk_ids: Chunk IDs that are relevant (ground truth).

        Returns:
            Recall score in range [0.0, 1.0].
            Returns 0.0 if ground_truth is empty.
        """
        if not ground_truth_chunk_ids:
            return 0.0

        retrieved_set = set(retrieved_chunk_ids)
        ground_truth_set = set(ground_truth_chunk_ids)
        intersection = retrieved_set & ground_truth_set

        return len(intersection) / len(ground_truth_set)


__all__ = ["ChunkRecall"]
