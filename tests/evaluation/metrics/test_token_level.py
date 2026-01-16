"""Tests for token-level metrics."""
from rag_evaluation_system.evaluation.metrics.token_level import SpanIoU, SpanPrecision, SpanRecall
from rag_evaluation_system.evaluation.metrics.token_level.utils import merge_overlapping_spans
from rag_evaluation_system.types import CharacterSpan, DocumentId


def test_merge_overlapping_spans():
    spans = [
        CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=50, text="x" * 50),
        CharacterSpan(doc_id=DocumentId("doc1"), start=30, end=80, text="x" * 50),
    ]
    merged = merge_overlapping_spans(spans)
    assert len(merged) == 1
    assert merged[0].start == 0
    assert merged[0].end == 80


def test_span_recall_partial():
    gt = [CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=100, text="x" * 100)]
    retrieved = [CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=50, text="x" * 50)]
    metric = SpanRecall()
    assert metric.calculate(retrieved, gt) == 0.5


def test_span_precision_partial():
    gt = [CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=50, text="x" * 50)]
    retrieved = [CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=100, text="x" * 100)]
    metric = SpanPrecision()
    assert metric.calculate(retrieved, gt) == 0.5


def test_span_iou_partial():
    gt = [CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=100, text="x" * 100)]
    retrieved = [CharacterSpan(doc_id=DocumentId("doc1"), start=50, end=150, text="x" * 100)]
    metric = SpanIoU()
    assert abs(metric.calculate(retrieved, gt) - 0.333) < 0.01
