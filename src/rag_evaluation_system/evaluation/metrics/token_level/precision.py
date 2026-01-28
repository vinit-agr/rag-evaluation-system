"""Token-level (span) precision metric implementation."""

from rag_evaluation_system.evaluation.metrics.base import TokenLevelMetric
from rag_evaluation_system.evaluation.metrics.token_level.utils import (
    calculate_overlap,
    merge_overlapping_spans,
)
from rag_evaluation_system.types import CharacterSpan


class SpanPrecision(TokenLevelMetric):
    """Token-level (character span) precision metric.

    Measures the proportion of retrieved characters that are actually
    relevant (covered by ground truth). A precision of 1.0 means all
    retrieved characters were relevant.

    Formula:
        precision = overlap_chars / total_retrieved_chars

    The metric merges overlapping spans before calculation to avoid
    double-counting characters.

    Example:
        Ground truth: characters 0-100 (100 chars)
        Retrieved: characters 50-150 (100 chars)
        Overlap: characters 50-100 = 50 chars
        Precision = 50 / 100 = 0.5
    """

    @property
    def name(self) -> str:
        """Return the metric name."""
        return "span_precision"

    def calculate(
        self,
        retrieved_spans: list[CharacterSpan],
        ground_truth_spans: list[CharacterSpan],
    ) -> float:
        """Calculate span precision.

        Args:
            retrieved_spans: Character spans from retrieved chunks.
            ground_truth_spans: Character spans that are relevant (ground truth).

        Returns:
            Precision score in range [0.0, 1.0].
            Returns 0.0 if retrieved is empty.
        """
        if not retrieved_spans:
            return 0.0

        # Merge retrieved spans and calculate total characters
        merged_retrieved = merge_overlapping_spans(retrieved_spans)
        total_retrieved_chars = sum(span.length for span in merged_retrieved)

        if total_retrieved_chars == 0:
            return 0.0

        # Calculate overlap
        overlap_chars = calculate_overlap(retrieved_spans, ground_truth_spans)

        # Cap at 1.0 to handle edge cases
        return min(1.0, overlap_chars / total_retrieved_chars)


__all__ = ["SpanPrecision"]
