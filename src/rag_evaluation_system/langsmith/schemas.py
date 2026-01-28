"""LangSmith dataset schema definitions for the RAG evaluation system.

This module defines the expected schemas for chunk-level and token-level
datasets stored in LangSmith. These serve as documentation for the data
format and can be used for validation.
"""

from typing import Final

# =============================================================================
# CHUNK-LEVEL DATASET SCHEMA
# =============================================================================

CHUNK_LEVEL_SCHEMA: Final[dict[str, object]] = {
    "name": "rag-eval-chunk-level",
    "description": "Ground truth for chunk-level RAG evaluation. "
    "Stores relevant chunk IDs for each query.",
    "example_schema": {
        "inputs": {
            "query": "string - The query/question text",
        },
        "outputs": {
            "relevant_chunk_ids": [
                "string - Chunk IDs in format: chunk_<12-char-hash>",
            ],
        },
        "metadata": {
            "source_docs": ["string - Document IDs where chunks originated"],
            "generation_model": "string - Model used to generate the example",
            "generation_type": "string - 'synthetic' or 'manual'",
            "persona": "string (optional) - User persona for the query",
            "difficulty": "string (optional) - Query difficulty level",
            "query_type": "string (optional) - Type of query",
        },
    },
    "example": {
        "inputs": {"query": "What are the benefits of RAG?"},
        "outputs": {
            "relevant_chunk_ids": [
                "chunk_a3f2b1c8d9e0",
                "chunk_7d9e4f2a1b3c",
            ]
        },
        "metadata": {
            "source_docs": ["rag_overview.md"],
            "generation_model": "gpt-4",
            "generation_type": "synthetic",
        },
    },
}

# =============================================================================
# TOKEN-LEVEL DATASET SCHEMA
# =============================================================================

TOKEN_LEVEL_SCHEMA: Final[dict[str, object]] = {
    "name": "rag-eval-token-level",
    "description": "Ground truth for token-level RAG evaluation. "
    "Stores relevant character spans (not chunk IDs) for each query. "
    "This format is chunker-independent, allowing fair comparison of different chunking strategies.",
    "example_schema": {
        "inputs": {
            "query": "string - The query/question text",
        },
        "outputs": {
            "relevant_spans": [
                {
                    "doc_id": "string - Document ID where the span is located",
                    "start": "integer - Starting character position (inclusive, 0-indexed)",
                    "end": "integer - Ending character position (exclusive)",
                    "text": "string - The actual text content of the span",
                },
            ],
        },
        "metadata": {
            "source_docs": ["string - Document IDs containing relevant content"],
            "generation_model": "string - Model used to generate the example",
            "generation_type": "string - 'synthetic' or 'manual'",
            "persona": "string (optional) - User persona for the query",
            "difficulty": "string (optional) - Query difficulty level",
            "query_type": "string (optional) - Type of query",
        },
    },
    "example": {
        "inputs": {"query": "What are the benefits of RAG?"},
        "outputs": {
            "relevant_spans": [
                {
                    "doc_id": "rag_overview.md",
                    "start": 1520,
                    "end": 1847,
                    "text": "RAG combines the benefits of retrieval systems with generative models...",
                },
                {
                    "doc_id": "rag_overview.md",
                    "start": 2103,
                    "end": 2298,
                    "text": "Key advantages include reduced hallucination and access to current information...",
                },
            ]
        },
        "metadata": {
            "source_docs": ["rag_overview.md"],
            "generation_model": "gpt-4",
            "generation_type": "synthetic",
        },
    },
}

__all__ = [
    "CHUNK_LEVEL_SCHEMA",
    "TOKEN_LEVEL_SCHEMA",
]
