"""Tests for token-level metrics and utilities.

Tests cover:
- merge_overlapping_spans utility function
- calculate_overlap utility function
- SpanRecall metric
- SpanPrecision metric
- SpanIoU metric
"""

import pytest

from rag_evaluation_system.evaluation.metrics.token_level import (
    SpanIoU,
    SpanPrecision,
    SpanRecall,
)
from rag_evaluation_system.evaluation.metrics.token_level.utils import (
    calculate_overlap,
    merge_overlapping_spans,
)
from rag_evaluation_system.types import CharacterSpan, DocumentId

# =============================================================================
# Test merge_overlapping_spans
# =============================================================================


class TestMergeOverlappingSpans:
    """Tests for the merge_overlapping_spans utility function."""

    def test_merge_no_overlap(self) -> None:
        """Test merging non-overlapping spans returns all spans unchanged."""
        spans = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=10, text="a" * 10),
            CharacterSpan(doc_id=DocumentId("doc.md"), start=20, end=30, text="b" * 10),
            CharacterSpan(doc_id=DocumentId("doc.md"), start=40, end=50, text="c" * 10),
        ]

        result = merge_overlapping_spans(spans)

        assert len(result) == 3
        # Check positions (text will be placeholder)
        starts = {s.start for s in result}
        ends = {s.end for s in result}
        assert starts == {0, 20, 40}
        assert ends == {10, 30, 50}

    def test_merge_overlapping_spans(self) -> None:
        """Test merging overlapping spans produces single merged span."""
        spans = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=10, text="a" * 10),
            CharacterSpan(doc_id=DocumentId("doc.md"), start=5, end=15, text="b" * 10),
        ]

        result = merge_overlapping_spans(spans)

        assert len(result) == 1
        assert result[0].start == 0
        assert result[0].end == 15
        assert result[0].length == 15

    def test_merge_adjacent_spans(self) -> None:
        """Test merging adjacent (touching) spans."""
        spans = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=10, text="a" * 10),
            CharacterSpan(doc_id=DocumentId("doc.md"), start=10, end=20, text="b" * 10),  # Adjacent
        ]

        result = merge_overlapping_spans(spans)

        # Adjacent spans (start <= current_end) should be merged
        assert len(result) == 1
        assert result[0].start == 0
        assert result[0].end == 20

    def test_merge_multiple_documents(self) -> None:
        """Test merging spans from multiple documents keeps them separate."""
        spans = [
            CharacterSpan(doc_id=DocumentId("doc1.md"), start=0, end=10, text="a" * 10),
            CharacterSpan(
                doc_id=DocumentId("doc1.md"), start=5, end=15, text="b" * 10
            ),  # Overlaps with first
            CharacterSpan(doc_id=DocumentId("doc2.md"), start=0, end=10, text="c" * 10),
            CharacterSpan(
                doc_id=DocumentId("doc2.md"), start=5, end=15, text="d" * 10
            ),  # Overlaps with third
        ]

        result = merge_overlapping_spans(spans)

        # Should have 2 merged spans, one for each document
        assert len(result) == 2
        doc_ids = {s.doc_id for s in result}
        assert doc_ids == {DocumentId("doc1.md"), DocumentId("doc2.md")}
        for span in result:
            assert span.start == 0
            assert span.end == 15

    def test_merge_empty_list(self) -> None:
        """Test merging empty list returns empty list."""
        result = merge_overlapping_spans([])

        assert result == []

    def test_merge_single_span(self) -> None:
        """Test merging single span returns that span."""
        spans = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=10, end=20, text="a" * 10),
        ]

        result = merge_overlapping_spans(spans)

        assert len(result) == 1
        assert result[0].start == 10
        assert result[0].end == 20

    def test_merge_contained_spans(self) -> None:
        """Test merging when one span contains another."""
        spans = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=100, text="a" * 100),
            CharacterSpan(
                doc_id=DocumentId("doc.md"), start=25, end=75, text="b" * 50
            ),  # Contained
        ]

        result = merge_overlapping_spans(spans)

        assert len(result) == 1
        assert result[0].start == 0
        assert result[0].end == 100

    def test_merge_multiple_overlapping_chain(self) -> None:
        """Test merging a chain of overlapping spans."""
        spans = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=10, text="a" * 10),
            CharacterSpan(doc_id=DocumentId("doc.md"), start=8, end=18, text="b" * 10),
            CharacterSpan(doc_id=DocumentId("doc.md"), start=16, end=26, text="c" * 10),
            CharacterSpan(doc_id=DocumentId("doc.md"), start=24, end=34, text="d" * 10),
        ]

        result = merge_overlapping_spans(spans)

        # All spans should merge into one
        assert len(result) == 1
        assert result[0].start == 0
        assert result[0].end == 34


# =============================================================================
# Test calculate_overlap
# =============================================================================


class TestCalculateOverlap:
    """Tests for the calculate_overlap utility function."""

    def test_calculate_overlap_basic(self) -> None:
        """Test basic overlap calculation."""
        spans_a = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=10, text="a" * 10),
        ]
        spans_b = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=5, end=15, text="b" * 10),
        ]

        result = calculate_overlap(spans_a, spans_b)

        # Overlap is from 5 to 10 = 5 characters
        assert result == 5

    def test_calculate_overlap_no_overlap(self) -> None:
        """Test overlap calculation with no overlap."""
        spans_a = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=10, text="a" * 10),
        ]
        spans_b = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=20, end=30, text="b" * 10),
        ]

        result = calculate_overlap(spans_a, spans_b)

        assert result == 0

    def test_calculate_overlap_complete_overlap(self) -> None:
        """Test overlap calculation with complete overlap."""
        spans_a = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=100, text="a" * 100),
        ]
        spans_b = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=25, end=75, text="b" * 50),
        ]

        result = calculate_overlap(spans_a, spans_b)

        # Complete overlap of the smaller span
        assert result == 50

    def test_calculate_overlap_empty_spans_a(self) -> None:
        """Test overlap calculation with empty first list."""
        spans_a: list[CharacterSpan] = []
        spans_b = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=10, text="a" * 10),
        ]

        result = calculate_overlap(spans_a, spans_b)

        assert result == 0

    def test_calculate_overlap_empty_spans_b(self) -> None:
        """Test overlap calculation with empty second list."""
        spans_a = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=10, text="a" * 10),
        ]
        spans_b: list[CharacterSpan] = []

        result = calculate_overlap(spans_a, spans_b)

        assert result == 0

    def test_calculate_overlap_both_empty(self) -> None:
        """Test overlap calculation with both lists empty."""
        result = calculate_overlap([], [])

        assert result == 0

    def test_calculate_overlap_multiple_spans(self) -> None:
        """Test overlap calculation with multiple spans in each collection."""
        spans_a = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=20, text="a" * 20),
            CharacterSpan(doc_id=DocumentId("doc.md"), start=50, end=70, text="b" * 20),
        ]
        spans_b = [
            CharacterSpan(
                doc_id=DocumentId("doc.md"), start=10, end=60, text="c" * 50
            ),  # Overlaps both
        ]

        result = calculate_overlap(spans_a, spans_b)

        # First overlap: 10-20 = 10 chars
        # Second overlap: 50-60 = 10 chars
        # Total = 20
        assert result == 20

    def test_calculate_overlap_different_documents(self) -> None:
        """Test overlap calculation ignores different documents."""
        spans_a = [
            CharacterSpan(doc_id=DocumentId("doc1.md"), start=0, end=10, text="a" * 10),
        ]
        spans_b = [
            CharacterSpan(
                doc_id=DocumentId("doc2.md"), start=0, end=10, text="b" * 10
            ),  # Same positions, different doc
        ]

        result = calculate_overlap(spans_a, spans_b)

        assert result == 0

    def test_calculate_overlap_merges_before_comparing(self) -> None:
        """Test that overlapping spans within a collection are merged first."""
        # Two overlapping spans in spans_a that together cover 0-15
        spans_a = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=10, text="a" * 10),
            CharacterSpan(doc_id=DocumentId("doc.md"), start=5, end=15, text="b" * 10),
        ]
        spans_b = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=10, end=20, text="c" * 10),
        ]

        result = calculate_overlap(spans_a, spans_b)

        # After merging spans_a: (0, 15)
        # Overlap with spans_b (10, 20): 10-15 = 5 chars
        assert result == 5


# =============================================================================
# Test SpanRecall
# =============================================================================


class TestSpanRecall:
    """Tests for the SpanRecall metric."""

    def test_span_recall_name(self) -> None:
        """Test that SpanRecall has the correct name."""
        metric = SpanRecall()
        assert metric.name == "span_recall"

    def test_span_recall_perfect_recall(self) -> None:
        """Test recall is 1.0 when all ground truth characters are retrieved."""
        metric = SpanRecall()
        ground_truth = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=100, text="a" * 100),
        ]
        retrieved = [
            CharacterSpan(
                doc_id=DocumentId("doc.md"), start=0, end=150, text="b" * 150
            ),  # Covers all of GT
        ]

        result = metric.calculate(retrieved, ground_truth)

        assert result == 1.0

    def test_span_recall_partial_overlap(self) -> None:
        """Test recall calculation with partial overlap."""
        metric = SpanRecall()
        ground_truth = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=100, text="a" * 100),
        ]
        retrieved = [
            CharacterSpan(
                doc_id=DocumentId("doc.md"), start=50, end=150, text="b" * 100
            ),  # Covers 50-100
        ]

        result = metric.calculate(retrieved, ground_truth)

        # 50 characters overlap out of 100 ground truth
        assert result == pytest.approx(0.5)

    def test_span_recall_no_overlap(self) -> None:
        """Test recall is 0.0 when there is no overlap."""
        metric = SpanRecall()
        ground_truth = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=100, text="a" * 100),
        ]
        retrieved = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=200, end=300, text="b" * 100),
        ]

        result = metric.calculate(retrieved, ground_truth)

        assert result == 0.0

    def test_span_recall_empty_ground_truth(self) -> None:
        """Test recall is 0.0 when ground truth is empty."""
        metric = SpanRecall()
        ground_truth: list[CharacterSpan] = []
        retrieved = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=50, text="a" * 50),
        ]

        result = metric.calculate(retrieved, ground_truth)

        assert result == 0.0

    def test_span_recall_empty_retrieved(self) -> None:
        """Test recall is 0.0 when retrieved is empty."""
        metric = SpanRecall()
        ground_truth = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=50, text="a" * 50),
        ]
        retrieved: list[CharacterSpan] = []

        result = metric.calculate(retrieved, ground_truth)

        assert result == 0.0

    def test_span_recall_multiple_spans(self) -> None:
        """Test recall with multiple ground truth and retrieved spans."""
        metric = SpanRecall()
        ground_truth = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=50, text="a" * 50),
            CharacterSpan(doc_id=DocumentId("doc.md"), start=100, end=150, text="b" * 50),
        ]  # Total: 100 chars
        retrieved = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=25, end=75, text="c" * 50),
            CharacterSpan(doc_id=DocumentId("doc.md"), start=125, end=175, text="d" * 50),
        ]

        result = metric.calculate(retrieved, ground_truth)

        # First overlap: 25-50 = 25 chars
        # Second overlap: 125-150 = 25 chars
        # Total overlap: 50 chars out of 100 ground truth
        assert result == pytest.approx(0.5)

    def test_span_recall_different_documents(self) -> None:
        """Test recall handles different documents correctly."""
        metric = SpanRecall()
        ground_truth = [
            CharacterSpan(doc_id=DocumentId("doc1.md"), start=0, end=100, text="a" * 100),
        ]
        retrieved = [
            CharacterSpan(
                doc_id=DocumentId("doc2.md"), start=0, end=100, text="b" * 100
            ),  # Different doc
        ]

        result = metric.calculate(retrieved, ground_truth)

        assert result == 0.0


# =============================================================================
# Test SpanPrecision
# =============================================================================


class TestSpanPrecision:
    """Tests for the SpanPrecision metric."""

    def test_span_precision_name(self) -> None:
        """Test that SpanPrecision has the correct name."""
        metric = SpanPrecision()
        assert metric.name == "span_precision"

    def test_span_precision_perfect_precision(self) -> None:
        """Test precision is 1.0 when all retrieved characters are relevant."""
        metric = SpanPrecision()
        ground_truth = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=200, text="a" * 200),
        ]
        retrieved = [
            CharacterSpan(
                doc_id=DocumentId("doc.md"), start=50, end=150, text="b" * 100
            ),  # Fully within GT
        ]

        result = metric.calculate(retrieved, ground_truth)

        assert result == 1.0

    def test_span_precision_partial_precision(self) -> None:
        """Test precision calculation with partial overlap."""
        metric = SpanPrecision()
        ground_truth = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=100, text="a" * 100),
        ]
        retrieved = [
            CharacterSpan(
                doc_id=DocumentId("doc.md"), start=50, end=150, text="b" * 100
            ),  # 50 chars relevant
        ]

        result = metric.calculate(retrieved, ground_truth)

        # 50 characters relevant out of 100 retrieved
        assert result == pytest.approx(0.5)

    def test_span_precision_no_precision(self) -> None:
        """Test precision is 0.0 when no retrieved characters are relevant."""
        metric = SpanPrecision()
        ground_truth = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=100, text="a" * 100),
        ]
        retrieved = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=200, end=300, text="b" * 100),
        ]

        result = metric.calculate(retrieved, ground_truth)

        assert result == 0.0

    def test_span_precision_empty_retrieved(self) -> None:
        """Test precision is 0.0 when retrieved is empty."""
        metric = SpanPrecision()
        ground_truth = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=50, text="a" * 50),
        ]
        retrieved: list[CharacterSpan] = []

        result = metric.calculate(retrieved, ground_truth)

        assert result == 0.0

    def test_span_precision_empty_ground_truth(self) -> None:
        """Test precision is 0.0 when ground truth is empty."""
        metric = SpanPrecision()
        ground_truth: list[CharacterSpan] = []
        retrieved = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=50, text="a" * 50),
        ]

        result = metric.calculate(retrieved, ground_truth)

        # No characters are relevant, so precision is 0
        assert result == 0.0

    def test_span_precision_multiple_spans(self) -> None:
        """Test precision with multiple retrieved spans."""
        metric = SpanPrecision()
        ground_truth = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=50, text="a" * 50),
        ]
        retrieved = [
            CharacterSpan(
                doc_id=DocumentId("doc.md"), start=0, end=25, text="b" * 25
            ),  # Fully relevant
            CharacterSpan(
                doc_id=DocumentId("doc.md"), start=50, end=100, text="c" * 50
            ),  # Not relevant
        ]

        result = metric.calculate(retrieved, ground_truth)

        # 25 relevant out of 75 retrieved
        assert result == pytest.approx(25 / 75)


# =============================================================================
# Test SpanIoU
# =============================================================================


class TestSpanIoU:
    """Tests for the SpanIoU metric."""

    def test_span_iou_name(self) -> None:
        """Test that SpanIoU has the correct name."""
        metric = SpanIoU()
        assert metric.name == "span_iou"

    def test_span_iou_perfect_match(self) -> None:
        """Test IoU is 1.0 when retrieved exactly matches ground truth."""
        metric = SpanIoU()
        spans = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=100, text="a" * 100),
        ]

        result = metric.calculate(spans, spans)

        assert result == 1.0

    def test_span_iou_partial_overlap(self) -> None:
        """Test IoU calculation with partial overlap."""
        metric = SpanIoU()
        ground_truth = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=100, text="a" * 100),
        ]
        retrieved = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=50, end=150, text="b" * 100),
        ]

        result = metric.calculate(retrieved, ground_truth)

        # Intersection: 50-100 = 50 chars
        # Union: 100 + 100 - 50 = 150 chars
        # IoU = 50 / 150 = 1/3
        assert result == pytest.approx(1 / 3)

    def test_span_iou_no_overlap(self) -> None:
        """Test IoU is 0.0 when there is no overlap."""
        metric = SpanIoU()
        ground_truth = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=100, text="a" * 100),
        ]
        retrieved = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=200, end=300, text="b" * 100),
        ]

        result = metric.calculate(retrieved, ground_truth)

        assert result == 0.0

    def test_span_iou_both_empty_lists(self) -> None:
        """Test IoU is 1.0 when both lists are empty (vacuously true)."""
        metric = SpanIoU()
        ground_truth: list[CharacterSpan] = []
        retrieved: list[CharacterSpan] = []

        result = metric.calculate(retrieved, ground_truth)

        assert result == 1.0

    def test_span_iou_empty_retrieved(self) -> None:
        """Test IoU is 0.0 when only retrieved is empty."""
        metric = SpanIoU()
        ground_truth = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=50, text="a" * 50),
        ]
        retrieved: list[CharacterSpan] = []

        result = metric.calculate(retrieved, ground_truth)

        assert result == 0.0

    def test_span_iou_empty_ground_truth(self) -> None:
        """Test IoU is 0.0 when only ground truth is empty."""
        metric = SpanIoU()
        ground_truth: list[CharacterSpan] = []
        retrieved = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=50, text="a" * 50),
        ]

        result = metric.calculate(retrieved, ground_truth)

        assert result == 0.0

    def test_span_iou_contained_span(self) -> None:
        """Test IoU when one span contains the other."""
        metric = SpanIoU()
        ground_truth = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=100, text="a" * 100),
        ]
        retrieved = [
            CharacterSpan(
                doc_id=DocumentId("doc.md"), start=25, end=75, text="b" * 50
            ),  # Contained in GT
        ]

        result = metric.calculate(retrieved, ground_truth)

        # Intersection: 50 chars (the entire retrieved span)
        # Union: 100 + 50 - 50 = 100 chars (just the GT since retrieved is contained)
        # IoU = 50 / 100 = 0.5
        assert result == pytest.approx(0.5)

    def test_span_iou_multiple_spans(self) -> None:
        """Test IoU with multiple spans in each collection."""
        metric = SpanIoU()
        ground_truth = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=50, text="a" * 50),
            CharacterSpan(doc_id=DocumentId("doc.md"), start=100, end=150, text="b" * 50),
        ]  # Total: 100 chars
        retrieved = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=25, end=125, text="c" * 100),
        ]  # Total: 100 chars

        result = metric.calculate(retrieved, ground_truth)

        # Intersection: (25-50) + (100-125) = 25 + 25 = 50 chars
        # Union: 100 + 100 - 50 = 150 chars
        # IoU = 50 / 150 = 1/3
        assert result == pytest.approx(1 / 3)

    def test_span_iou_known_example(self) -> None:
        """Test IoU with a documented example from the docstring."""
        metric = SpanIoU()
        # From docstring:
        # Ground truth: characters 0-100 (100 chars)
        # Retrieved: characters 50-150 (100 chars)
        # Intersection: characters 50-100 = 50 chars
        # Union: 100 + 100 - 50 = 150 chars
        # IoU = 50 / 150 = 0.333

        ground_truth = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=0, end=100, text="a" * 100),
        ]
        retrieved = [
            CharacterSpan(doc_id=DocumentId("doc.md"), start=50, end=150, text="b" * 100),
        ]

        result = metric.calculate(retrieved, ground_truth)

        assert result == pytest.approx(50 / 150, rel=1e-3)
