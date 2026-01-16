"""Base class for synthetic data generators."""
from abc import ABC
from typing import Any

from rag_evaluation_system.types import Corpus


class SyntheticDatasetGenerator(ABC):
    """Base class for synthetic data generation."""

    def __init__(self, llm_client: Any, corpus: Corpus):
        self._llm = llm_client
        self._corpus = corpus

    @property
    def corpus(self) -> Corpus:
        return self._corpus
