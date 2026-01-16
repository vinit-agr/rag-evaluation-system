"""Span IoU (Intersection over Union) metric."""
from rag_evaluation_system.types import CharacterSpan
from ..base import TokenLevelMetric
from .utils import calculate_overlap, merge_overlapping_spans


class SpanIoU(TokenLevelMetric):
    """Intersection over Union of character spans."""

    @property
    def name(self) -> str:
        return "span_iou"

    def calculate(
        self,
        retrieved_spans: list[CharacterSpan],
        ground_truth_spans: list[CharacterSpan],
    ) -> float:
        if not retrieved_spans and not ground_truth_spans:
            return 1.0
        if not retrieved_spans or not ground_truth_spans:
            return 0.0

        merged_retrieved = merge_overlapping_spans(retrieved_spans)
        merged_gt = merge_overlapping_spans(ground_truth_spans)

        intersection = calculate_overlap(retrieved_spans, ground_truth_spans)
        total_retrieved = sum(span.length for span in merged_retrieved)
        total_gt = sum(span.length for span in merged_gt)
        union = total_retrieved + total_gt - intersection

        return intersection / union if union > 0 else 0.0
