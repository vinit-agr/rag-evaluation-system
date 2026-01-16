"""Chunk-level precision metric implementation."""

from rag_evaluation_system.evaluation.metrics.base import ChunkLevelMetric
from rag_evaluation_system.types import ChunkId


class ChunkPrecision(ChunkLevelMetric):
    """Chunk-level precision metric.

    Measures the proportion of retrieved chunks that are actually relevant.
    A precision of 1.0 means all retrieved chunks were relevant.

    Formula:
        precision = |retrieved ∩ ground_truth| / |retrieved|

    Example:
        If ground_truth = {A, B, C} and retrieved = {A, B, D}:
        intersection = {A, B}
        precision = 2 / 3 = 0.667
    """

    @property
    def name(self) -> str:
        """Return the metric name."""
        return "chunk_precision"

    def calculate(
        self,
        retrieved_chunk_ids: list[ChunkId],
        ground_truth_chunk_ids: list[ChunkId],
    ) -> float:
        """Calculate chunk precision.

        Args:
            retrieved_chunk_ids: Chunk IDs returned by the retrieval system.
            ground_truth_chunk_ids: Chunk IDs that are relevant (ground truth).

        Returns:
            Precision score in range [0.0, 1.0].
            Returns 0.0 if retrieved is empty.
        """
        if not retrieved_chunk_ids:
            return 0.0

        retrieved_set = set(retrieved_chunk_ids)
        ground_truth_set = set(ground_truth_chunk_ids)
        intersection = retrieved_set & ground_truth_set

        return len(intersection) / len(retrieved_set)


__all__ = ["ChunkPrecision"]
