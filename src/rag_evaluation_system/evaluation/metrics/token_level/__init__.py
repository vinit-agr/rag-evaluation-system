"""Token-level metrics (SpanRecall, SpanPrecision, SpanIoU).

These metrics operate on character spans, enabling chunker-independent
evaluation. This allows fair comparison across different chunking strategies.
Overlapping spans are merged before calculation to prevent sliding-window
chunkers from inflating metrics.
"""

from rag_evaluation_system.evaluation.metrics.token_level.iou import SpanIoU
from rag_evaluation_system.evaluation.metrics.token_level.precision import (
    SpanPrecision,
)
from rag_evaluation_system.evaluation.metrics.token_level.recall import SpanRecall
from rag_evaluation_system.evaluation.metrics.token_level.utils import (
    calculate_overlap,
    merge_overlapping_spans,
)

__all__ = [
    "SpanIoU",
    "SpanPrecision",
    "SpanRecall",
    "calculate_overlap",
    "merge_overlapping_spans",
]
