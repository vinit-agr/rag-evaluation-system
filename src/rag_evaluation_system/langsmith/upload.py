"""LangSmith upload utilities for creating datasets.

This module provides functions for uploading chunk-level and token-level
ground truth datasets to LangSmith.
"""

import logging
from typing import Any

from rag_evaluation_system.langsmith.client import get_client
from rag_evaluation_system.types import (
    ChunkLevelGroundTruth,
    TokenLevelGroundTruth,
)

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_LEVEL_DATASET_NAME = "rag-eval-chunk-level"
DEFAULT_TOKEN_LEVEL_DATASET_NAME = "rag-eval-token-level"


def upload_chunk_level_dataset(
    ground_truth: list[ChunkLevelGroundTruth],
    dataset_name: str | None = None,
) -> str:
    """Upload chunk-level ground truth data to LangSmith.

    Creates a new dataset in LangSmith and populates it with the provided
    ground truth examples. Each example contains a query and the list of
    relevant chunk IDs.

    Args:
        ground_truth: List of ChunkLevelGroundTruth objects to upload.
        dataset_name: Name for the dataset. Defaults to "rag-eval-chunk-level".

    Returns:
        The name of the created dataset.

    Raises:
        langsmith.utils.LangSmithError: If there's an error communicating with LangSmith.

    Example:
        >>> from rag_evaluation_system.types import ChunkLevelGroundTruth, Query, ChunkId
        >>> gt = ChunkLevelGroundTruth(
        ...     query=Query(id=QueryId("q1"), text=QueryText("What is RAG?")),
        ...     relevant_chunk_ids=[ChunkId("chunk_abc123")],
        ... )
        >>> dataset_name = upload_chunk_level_dataset([gt])
        >>> print(f"Created dataset: {dataset_name}")
    """
    if dataset_name is None:
        dataset_name = DEFAULT_CHUNK_LEVEL_DATASET_NAME

    client = get_client()

    # Create the dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Ground truth for chunk-level RAG evaluation. "
        "Contains queries mapped to relevant chunk IDs.",
    )

    # Prepare examples for batch upload
    inputs_list: list[dict[str, Any]] = []
    outputs_list: list[dict[str, Any]] = []
    metadata_list: list[dict[str, Any]] = []

    for gt in ground_truth:
        # Inputs: just the query text
        inputs_list.append({"query": str(gt.query.text)})

        # Outputs: list of chunk IDs as strings
        outputs_list.append({"relevant_chunk_ids": [str(cid) for cid in gt.relevant_chunk_ids]})

        # Metadata: from the query's metadata (top-level for LangSmith)
        metadata_list.append(gt.query.metadata)

    # Create examples in batch
    client.create_examples(
        inputs=inputs_list,
        outputs=outputs_list,
        metadata=metadata_list,
        dataset_id=dataset.id,
    )

    logger.info(
        "Uploaded %d chunk-level examples to dataset '%s'",
        len(ground_truth),
        dataset_name,
    )

    return dataset_name


def upload_token_level_dataset(
    ground_truth: list[TokenLevelGroundTruth],
    dataset_name: str | None = None,
) -> str:
    """Upload token-level ground truth data to LangSmith.

    Creates a new dataset in LangSmith and populates it with the provided
    ground truth examples. Each example contains a query and the list of
    relevant character spans (doc_id, start, end, text).

    Args:
        ground_truth: List of TokenLevelGroundTruth objects to upload.
        dataset_name: Name for the dataset. Defaults to "rag-eval-token-level".

    Returns:
        The name of the created dataset.

    Raises:
        langsmith.utils.LangSmithError: If there's an error communicating with LangSmith.

    Example:
        >>> from rag_evaluation_system.types import (
        ...     TokenLevelGroundTruth,
        ...     Query,
        ...     CharacterSpan,
        ...     DocumentId,
        ... )
        >>> gt = TokenLevelGroundTruth(
        ...     query=Query(id=QueryId("q1"), text=QueryText("What is RAG?")),
        ...     relevant_spans=[
        ...         CharacterSpan(
        ...             doc_id=DocumentId("doc.md"),
        ...             start=100,
        ...             end=200,
        ...             text="RAG combines retrieval...",
        ...         )
        ...     ],
        ... )
        >>> dataset_name = upload_token_level_dataset([gt])
        >>> print(f"Created dataset: {dataset_name}")
    """
    if dataset_name is None:
        dataset_name = DEFAULT_TOKEN_LEVEL_DATASET_NAME

    client = get_client()

    # Create the dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Ground truth for token-level RAG evaluation. "
        "Contains queries mapped to relevant character spans (doc_id, start, end, text). "
        "This format is chunker-independent for fair comparison of chunking strategies.",
    )

    # Prepare examples for batch upload
    inputs_list: list[dict[str, Any]] = []
    outputs_list: list[dict[str, Any]] = []
    metadata_list: list[dict[str, Any]] = []

    for gt in ground_truth:
        # Inputs: just the query text
        inputs_list.append({"query": str(gt.query.text)})

        # Outputs: list of span dictionaries
        spans_data = [
            {
                "doc_id": str(span.doc_id),
                "start": span.start,
                "end": span.end,
                "text": span.text,
            }
            for span in gt.relevant_spans
        ]
        outputs_list.append({"relevant_spans": spans_data})

        # Metadata: from the query's metadata (top-level for LangSmith)
        metadata_list.append(gt.query.metadata)

    # Create examples in batch
    client.create_examples(
        inputs=inputs_list,
        outputs=outputs_list,
        metadata=metadata_list,
        dataset_id=dataset.id,
    )

    logger.info(
        "Uploaded %d token-level examples to dataset '%s'",
        len(ground_truth),
        dataset_name,
    )

    return dataset_name


__all__ = [
    "upload_chunk_level_dataset",
    "upload_token_level_dataset",
]
