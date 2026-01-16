"""Base classes for evaluation metrics."""
from abc import ABC, abstractmethod

from rag_evaluation_system.types import CharacterSpan, ChunkId


class ChunkLevelMetric(ABC):
    """Base class for chunk-level metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def calculate(
        self,
        retrieved_chunk_ids: list[ChunkId],
        ground_truth_chunk_ids: list[ChunkId],
    ) -> float:
        raise NotImplementedError


class TokenLevelMetric(ABC):
    """Base class for token-level (character span) metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def calculate(
        self,
        retrieved_spans: list[CharacterSpan],
        ground_truth_spans: list[CharacterSpan],
    ) -> float:
        raise NotImplementedError
