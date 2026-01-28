"""Vector store interfaces and implementations.

This module provides the abstract VectorStore interface and concrete implementations
for storing and searching embeddings using various backends.

Available Vector Stores:
    - ChromaVectorStore: Uses ChromaDB (requires `chroma` extra)

Example:
    >>> from rag_evaluation_system.vector_stores import VectorStore, ChromaVectorStore
    >>> store: VectorStore = ChromaVectorStore()
    >>> store.add(chunks, embeddings)
    >>> results = store.search(query_embedding, k=5)
"""

from rag_evaluation_system.vector_stores.base import VectorStore

__all__ = ["VectorStore"]

# Conditionally export ChromaVectorStore if chromadb is available
try:
    from rag_evaluation_system.vector_stores.chroma import ChromaVectorStore as ChromaVectorStore

    __all__.append("ChromaVectorStore")
except ImportError:
    pass
