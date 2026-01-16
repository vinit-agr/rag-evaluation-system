"""Chunk precision metric."""
from rag_evaluation_system.types import ChunkId
from ..base import ChunkLevelMetric


class ChunkPrecision(ChunkLevelMetric):
    """What fraction of retrieved chunks were relevant?"""

    @property
    def name(self) -> str:
        return "chunk_precision"

    def calculate(self, retrieved_chunk_ids: list[ChunkId], ground_truth_chunk_ids: list[ChunkId]) -> float:
        if not retrieved_chunk_ids:
            return 0.0

        retrieved_set = set(retrieved_chunk_ids)
        ground_truth_set = set(ground_truth_chunk_ids)
        intersection = retrieved_set & ground_truth_set
        return len(intersection) / len(retrieved_set)
