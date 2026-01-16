"""Tests for chunk-level metrics."""
from rag_evaluation_system.evaluation.metrics.chunk_level import ChunkF1, ChunkPrecision, ChunkRecall
from rag_evaluation_system.types import ChunkId


def test_chunk_recall():
    metric = ChunkRecall()
    assert metric.calculate([ChunkId("chunk_a")], [ChunkId("chunk_a"), ChunkId("chunk_b")]) == 0.5


def test_chunk_precision():
    metric = ChunkPrecision()
    assert metric.calculate([ChunkId("chunk_a"), ChunkId("chunk_b")], [ChunkId("chunk_b")]) == 0.5


def test_chunk_f1():
    metric = ChunkF1()
    score = metric.calculate([ChunkId("chunk_a")], [ChunkId("chunk_a"), ChunkId("chunk_b")])
    assert abs(score - (2 * (1.0 * 0.5) / 1.5)) < 0.0001
