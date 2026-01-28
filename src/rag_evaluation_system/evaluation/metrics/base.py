"""Abstract base classes for evaluation metrics.

This module defines the abstract interfaces for both chunk-level and token-level
metrics. All concrete metric implementations must inherit from these base classes.
"""

from abc import ABC, abstractmethod

from rag_evaluation_system.types import CharacterSpan, ChunkId


class ChunkLevelMetric(ABC):
    """Abstract base class for chunk-level evaluation metrics.

    Chunk-level metrics operate on sets of chunk IDs, comparing retrieved
    chunks against ground truth chunks. This is simpler but ties evaluation
    to a specific chunking strategy.

    Subclasses must implement:
        - name: A unique identifier for the metric (e.g., "chunk_recall").
        - calculate: The metric computation logic.

    Example:
        class ChunkRecall(ChunkLevelMetric):
            @property
            def name(self) -> str:
                return "chunk_recall"

            def calculate(
                self,
                retrieved_chunk_ids: list[ChunkId],
                ground_truth_chunk_ids: list[ChunkId],
            ) -> float:
                # Implementation here
                ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name identifier for this metric.

        Returns:
            A string identifier (e.g., "chunk_recall", "chunk_precision").
        """
        ...

    @abstractmethod
    def calculate(
        self,
        retrieved_chunk_ids: list[ChunkId],
        ground_truth_chunk_ids: list[ChunkId],
    ) -> float:
        """Calculate the metric value.

        Args:
            retrieved_chunk_ids: List of chunk IDs returned by the retrieval system.
            ground_truth_chunk_ids: List of chunk IDs that are relevant (ground truth).

        Returns:
            A float value typically in the range [0.0, 1.0].
        """
        ...


class TokenLevelMetric(ABC):
    """Abstract base class for token-level (character span) evaluation metrics.

    Token-level metrics operate on character spans, enabling chunker-independent
    evaluation. This allows fair comparison across different chunking strategies.

    Subclasses must implement:
        - name: A unique identifier for the metric (e.g., "span_recall").
        - calculate: The metric computation logic.

    Note:
        Implementations should typically merge overlapping spans before
        calculation to avoid double-counting characters.

    Example:
        class SpanRecall(TokenLevelMetric):
            @property
            def name(self) -> str:
                return "span_recall"

            def calculate(
                self,
                retrieved_spans: list[CharacterSpan],
                ground_truth_spans: list[CharacterSpan],
            ) -> float:
                # Implementation here
                ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name identifier for this metric.

        Returns:
            A string identifier (e.g., "span_recall", "span_precision").
        """
        ...

    @abstractmethod
    def calculate(
        self,
        retrieved_spans: list[CharacterSpan],
        ground_truth_spans: list[CharacterSpan],
    ) -> float:
        """Calculate the metric value.

        Args:
            retrieved_spans: List of character spans from retrieved chunks.
            ground_truth_spans: List of character spans that are relevant (ground truth).

        Returns:
            A float value typically in the range [0.0, 1.0].
        """
        ...


__all__ = [
    "ChunkLevelMetric",
    "TokenLevelMetric",
]
