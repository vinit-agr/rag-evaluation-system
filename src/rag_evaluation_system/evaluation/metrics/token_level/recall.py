"""Token-level (span) recall metric implementation."""

from rag_evaluation_system.evaluation.metrics.base import TokenLevelMetric
from rag_evaluation_system.evaluation.metrics.token_level.utils import (
    calculate_overlap,
    merge_overlapping_spans,
)
from rag_evaluation_system.types import CharacterSpan


class SpanRecall(TokenLevelMetric):
    """Token-level (character span) recall metric.

    Measures the proportion of ground truth characters that were covered
    by retrieved spans. A recall of 1.0 means all relevant characters
    were retrieved.

    Formula:
        recall = overlap_chars / total_gt_chars

    The metric merges overlapping spans before calculation to avoid
    double-counting characters.

    Example:
        Ground truth: characters 0-100 (100 chars)
        Retrieved: characters 50-150 (100 chars)
        Overlap: characters 50-100 = 50 chars
        Recall = 50 / 100 = 0.5
    """

    @property
    def name(self) -> str:
        """Return the metric name."""
        return "span_recall"

    def calculate(
        self,
        retrieved_spans: list[CharacterSpan],
        ground_truth_spans: list[CharacterSpan],
    ) -> float:
        """Calculate span recall.

        Args:
            retrieved_spans: Character spans from retrieved chunks.
            ground_truth_spans: Character spans that are relevant (ground truth).

        Returns:
            Recall score in range [0.0, 1.0].
            Returns 0.0 if ground_truth is empty.
        """
        if not ground_truth_spans:
            return 0.0

        # Merge ground truth spans and calculate total characters
        merged_gt = merge_overlapping_spans(ground_truth_spans)
        total_gt_chars = sum(span.length for span in merged_gt)

        if total_gt_chars == 0:
            return 0.0

        # Calculate overlap
        overlap_chars = calculate_overlap(retrieved_spans, ground_truth_spans)

        # Cap at 1.0 to handle edge cases
        return min(1.0, overlap_chars / total_gt_chars)


__all__ = ["SpanRecall"]
