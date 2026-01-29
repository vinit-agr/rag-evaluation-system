"""LangSmith integration for dataset management.

This module provides utilities for uploading and loading RAG evaluation
datasets to/from LangSmith. It supports both chunk-level and token-level
evaluation paradigms.

Example usage:

    # Upload chunk-level ground truth
    from rag_evaluation_system.langsmith import upload_chunk_level_dataset
    dataset_name = upload_chunk_level_dataset(ground_truths, "my-dataset")

    # Load token-level ground truth
    from rag_evaluation_system.langsmith import load_token_level_dataset
    ground_truths = load_token_level_dataset("my-token-level-dataset")
"""

from rag_evaluation_system.langsmith.client import (
    get_client,
    load_chunk_level_dataset,
    load_token_level_dataset,
)
from rag_evaluation_system.langsmith.schemas import (
    CHUNK_LEVEL_SCHEMA,
    TOKEN_LEVEL_SCHEMA,
)
from rag_evaluation_system.langsmith.upload import (
    upload_chunk_level_dataset,
    upload_token_level_dataset,
)

__all__ = [
    "CHUNK_LEVEL_SCHEMA",
    "TOKEN_LEVEL_SCHEMA",
    "get_client",
    "load_chunk_level_dataset",
    "load_token_level_dataset",
    "upload_chunk_level_dataset",
    "upload_token_level_dataset",
]
