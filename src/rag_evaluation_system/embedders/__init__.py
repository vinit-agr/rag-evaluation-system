"""Embedder interfaces and implementations.

This module provides the abstract Embedder interface and concrete implementations
for generating text embeddings using various backends.

Available Embedders:
    - OpenAIEmbedder: Uses OpenAI's embedding API (requires `openai` extra)
    - SentenceTransformerEmbedder: Uses sentence-transformers (requires `sentence-transformers` extra)

Example:
    >>> from rag_evaluation_system.embedders import Embedder, OpenAIEmbedder
    >>> embedder: Embedder = OpenAIEmbedder()
    >>> embedding = embedder.embed_query("What is machine learning?")
"""

from rag_evaluation_system.embedders.base import Embedder

__all__ = ["Embedder"]

# Conditionally export OpenAIEmbedder if openai is available
try:
    from rag_evaluation_system.embedders.openai import OpenAIEmbedder as OpenAIEmbedder

    __all__.append("OpenAIEmbedder")
except ImportError:
    pass

# Conditionally export SentenceTransformerEmbedder if sentence-transformers is available
try:
    from rag_evaluation_system.embedders.sentence_transformers import (
        SentenceTransformerEmbedder as SentenceTransformerEmbedder,
    )

    __all__.append("SentenceTransformerEmbedder")
except ImportError:
    pass
