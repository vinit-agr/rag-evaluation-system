"""Span recall metric."""
from rag_evaluation_system.types import CharacterSpan
from ..base import TokenLevelMetric
from .utils import calculate_overlap, merge_overlapping_spans


class SpanRecall(TokenLevelMetric):
    """What fraction of ground truth characters were retrieved?"""

    @property
    def name(self) -> str:
        return "span_recall"

    def calculate(
        self,
        retrieved_spans: list[CharacterSpan],
        ground_truth_spans: list[CharacterSpan],
    ) -> float:
        if not ground_truth_spans:
            return 0.0

        merged_gt = merge_overlapping_spans(ground_truth_spans)
        total_gt_chars = sum(span.length for span in merged_gt)
        overlap_chars = calculate_overlap(retrieved_spans, ground_truth_spans)

        return min(overlap_chars / total_gt_chars, 1.0)
