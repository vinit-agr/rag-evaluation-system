"""Span precision metric."""
from rag_evaluation_system.types import CharacterSpan
from ..base import TokenLevelMetric
from .utils import calculate_overlap, merge_overlapping_spans


class SpanPrecision(TokenLevelMetric):
    """What fraction of retrieved characters were relevant?"""

    @property
    def name(self) -> str:
        return "span_precision"

    def calculate(
        self,
        retrieved_spans: list[CharacterSpan],
        ground_truth_spans: list[CharacterSpan],
    ) -> float:
        if not retrieved_spans:
            return 0.0

        merged_retrieved = merge_overlapping_spans(retrieved_spans)
        total_ret_chars = sum(span.length for span in merged_retrieved)
        overlap_chars = calculate_overlap(retrieved_spans, ground_truth_spans)

        return min(overlap_chars / total_ret_chars, 1.0)
