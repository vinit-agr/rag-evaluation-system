"""Base class for synthetic data generators.

This module defines the abstract base class for all synthetic data generators
in the RAG evaluation system.
"""

from typing import Any

from rag_evaluation_system.types.documents import Corpus


class SyntheticDatasetGenerator:
    """Base class for synthetic dataset generation.

    Provides common functionality for generating synthetic QA pairs
    for RAG evaluation. Subclasses implement specific strategies for
    chunk-level or token-level evaluation.

    Attributes:
        _llm: The LLM client used for generating queries and excerpts.
            Expected to support an OpenAI-compatible interface.
        _corpus: The document corpus to generate data from.
    """

    def __init__(self, llm_client: Any, corpus: Corpus) -> None:
        """Initialize the generator with an LLM client and corpus.

        Args:
            llm_client: An LLM client with an OpenAI-compatible interface.
                Should support chat completions with system/user messages.
            corpus: The document corpus to generate synthetic data from.
        """
        self._llm = llm_client
        self._corpus = corpus

    @property
    def corpus(self) -> Corpus:
        """Return the corpus used for synthetic data generation.

        Returns:
            The Corpus instance containing all source documents.
        """
        return self._corpus


__all__ = [
    "SyntheticDatasetGenerator",
]
