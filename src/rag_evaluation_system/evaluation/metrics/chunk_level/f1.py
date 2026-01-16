"""Chunk F1 metric."""
from rag_evaluation_system.types import ChunkId
from ..base import ChunkLevelMetric
from .precision import ChunkPrecision
from .recall import ChunkRecall


class ChunkF1(ChunkLevelMetric):
    """Harmonic mean of chunk precision and recall."""

    def __init__(self) -> None:
        self._recall = ChunkRecall()
        self._precision = ChunkPrecision()

    @property
    def name(self) -> str:
        return "chunk_f1"

    def calculate(self, retrieved_chunk_ids: list[ChunkId], ground_truth_chunk_ids: list[ChunkId]) -> float:
        recall = self._recall.calculate(retrieved_chunk_ids, ground_truth_chunk_ids)
        precision = self._precision.calculate(retrieved_chunk_ids, ground_truth_chunk_ids)

        if recall + precision == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)
