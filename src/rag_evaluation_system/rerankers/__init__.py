"""Reranker interfaces and implementations.

This module provides the abstract Reranker interface and concrete implementations
for reranking retrieved chunks based on relevance to a query.

Available Rerankers:
    - CohereReranker: Uses Cohere's Rerank API (requires `cohere` extra)

Example:
    >>> from rag_evaluation_system.rerankers import Reranker, CohereReranker
    >>> reranker: Reranker = CohereReranker()
    >>> reranked_chunks = reranker.rerank("What is machine learning?", chunks)
"""

from rag_evaluation_system.rerankers.base import Reranker

__all__ = ["Reranker"]

# Conditionally export CohereReranker if cohere is available
try:
    from rag_evaluation_system.rerankers.cohere import CohereReranker as CohereReranker

    __all__.append("CohereReranker")
except ImportError:
    pass
