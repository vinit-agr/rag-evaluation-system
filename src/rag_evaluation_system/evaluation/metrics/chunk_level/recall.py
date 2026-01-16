"""Chunk recall metric."""
from rag_evaluation_system.types import ChunkId
from ..base import ChunkLevelMetric


class ChunkRecall(ChunkLevelMetric):
    """What fraction of relevant chunks were retrieved?"""

    @property
    def name(self) -> str:
        return "chunk_recall"

    def calculate(self, retrieved_chunk_ids: list[ChunkId], ground_truth_chunk_ids: list[ChunkId]) -> float:
        if not ground_truth_chunk_ids:
            return 0.0

        retrieved_set = set(retrieved_chunk_ids)
        ground_truth_set = set(ground_truth_chunk_ids)
        intersection = retrieved_set & ground_truth_set
        return len(intersection) / len(ground_truth_set)
