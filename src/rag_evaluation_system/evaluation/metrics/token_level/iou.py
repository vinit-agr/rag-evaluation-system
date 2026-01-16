"""Token-level (span) Intersection over Union (IoU) metric implementation."""

from rag_evaluation_system.evaluation.metrics.base import TokenLevelMetric
from rag_evaluation_system.evaluation.metrics.token_level.utils import (
    calculate_overlap,
    merge_overlapping_spans,
)
from rag_evaluation_system.types import CharacterSpan


class SpanIoU(TokenLevelMetric):
    """Token-level (character span) Intersection over Union metric.

    Measures the overlap between retrieved and ground truth spans as a
    proportion of their union. IoU provides a balanced view that penalizes
    both missing relevant content and retrieving irrelevant content.

    Formula:
        IoU = intersection / union
        where union = total_retrieved + total_gt - intersection

    The metric merges overlapping spans before calculation to avoid
    double-counting characters.

    Example:
        Ground truth: characters 0-100 (100 chars)
        Retrieved: characters 50-150 (100 chars)
        Intersection: characters 50-100 = 50 chars
        Union: 100 + 100 - 50 = 150 chars
        IoU = 50 / 150 = 0.333
    """

    @property
    def name(self) -> str:
        """Return the metric name."""
        return "span_iou"

    def calculate(
        self,
        retrieved_spans: list[CharacterSpan],
        ground_truth_spans: list[CharacterSpan],
    ) -> float:
        """Calculate span IoU.

        Args:
            retrieved_spans: Character spans from retrieved chunks.
            ground_truth_spans: Character spans that are relevant (ground truth).

        Returns:
            IoU score in range [0.0, 1.0].
            Returns 1.0 if both are empty (vacuously true).
            Returns 0.0 if exactly one is empty.
        """
        # Merge spans and calculate totals
        merged_retrieved = merge_overlapping_spans(retrieved_spans)
        merged_gt = merge_overlapping_spans(ground_truth_spans)

        total_retrieved = sum(span.length for span in merged_retrieved)
        total_gt = sum(span.length for span in merged_gt)

        # Handle edge cases
        if total_retrieved == 0 and total_gt == 0:
            return 1.0  # Both empty - vacuously true
        if total_retrieved == 0 or total_gt == 0:
            return 0.0  # One is empty, no overlap possible

        # Calculate intersection
        intersection = calculate_overlap(retrieved_spans, ground_truth_spans)

        # Calculate union: |A| + |B| - |A ∩ B|
        union = total_retrieved + total_gt - intersection

        if union == 0:
            return 0.0

        return intersection / union


__all__ = ["SpanIoU"]
