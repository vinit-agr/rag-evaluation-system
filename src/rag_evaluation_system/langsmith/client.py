"""LangSmith client utilities for loading datasets.

This module provides functions for connecting to LangSmith and loading
chunk-level and token-level ground truth datasets.
"""

import logging
import uuid
from typing import Any

from langsmith import Client

from rag_evaluation_system.types import (
    CharacterSpan,
    ChunkId,
    ChunkLevelGroundTruth,
    DocumentId,
    Query,
    QueryId,
    QueryText,
    TokenLevelGroundTruth,
)

logger = logging.getLogger(__name__)


def get_client() -> Client:
    """Get a LangSmith client instance.

    The client uses the LANGCHAIN_API_KEY environment variable for authentication.
    Make sure this environment variable is set before calling this function.

    Returns:
        A LangSmith Client instance configured with the API key from environment.

    Raises:
        langsmith.utils.LangSmithError: If the API key is not set or invalid.
    """
    return Client()


def load_chunk_level_dataset(dataset_name: str) -> list[ChunkLevelGroundTruth]:
    """Load a chunk-level ground truth dataset from LangSmith.

    Retrieves all examples from the specified dataset and converts them
    into ChunkLevelGroundTruth objects.

    Args:
        dataset_name: The name of the LangSmith dataset to load.

    Returns:
        A list of ChunkLevelGroundTruth objects, one per example in the dataset.

    Raises:
        langsmith.utils.LangSmithNotFoundError: If the dataset doesn't exist.
        KeyError: If an example is missing required fields (query, relevant_chunk_ids).
        ValueError: If the data format is invalid.

    Example:
        >>> ground_truths = load_chunk_level_dataset("my-rag-eval-chunk-level")
        >>> for gt in ground_truths:
        ...     print(f"Query: {gt.query.text}")
        ...     print(f"Relevant chunks: {gt.relevant_chunk_ids}")
    """
    client = get_client()
    examples = client.list_examples(dataset_name=dataset_name)

    ground_truths: list[ChunkLevelGroundTruth] = []

    for example in examples:
        inputs = example.inputs
        outputs = example.outputs

        if inputs is None:
            raise ValueError(f"Example {example.id} has no inputs")
        if outputs is None:
            raise ValueError(f"Example {example.id} has no outputs")

        # Extract query from inputs
        query_text = inputs.get("query")
        if query_text is None:
            raise KeyError(f"Example {example.id} is missing 'query' in inputs")

        # Extract chunk IDs from outputs
        chunk_ids_raw = outputs.get("relevant_chunk_ids")
        if chunk_ids_raw is None:
            raise KeyError(f"Example {example.id} is missing 'relevant_chunk_ids' in outputs")

        # Convert to typed values
        chunk_ids = [ChunkId(str(cid)) for cid in chunk_ids_raw]

        # Get metadata - use example.metadata if available, otherwise try inputs metadata
        metadata: dict[str, Any] = {}
        if example.metadata is not None:
            metadata = dict(example.metadata)

        # Create Query object - use example ID as query ID if available
        query_id = str(example.id) if example.id else str(uuid.uuid4())
        query = Query(
            id=QueryId(query_id),
            text=QueryText(str(query_text)),
            metadata=metadata,
        )

        ground_truth = ChunkLevelGroundTruth(
            query=query,
            relevant_chunk_ids=chunk_ids,
        )
        ground_truths.append(ground_truth)

    logger.info(
        "Loaded %d chunk-level ground truth examples from dataset '%s'",
        len(ground_truths),
        dataset_name,
    )

    return ground_truths


def load_token_level_dataset(dataset_name: str) -> list[TokenLevelGroundTruth]:
    """Load a token-level ground truth dataset from LangSmith.

    Retrieves all examples from the specified dataset and converts them
    into TokenLevelGroundTruth objects with CharacterSpan representations.

    Args:
        dataset_name: The name of the LangSmith dataset to load.

    Returns:
        A list of TokenLevelGroundTruth objects, one per example in the dataset.

    Raises:
        langsmith.utils.LangSmithNotFoundError: If the dataset doesn't exist.
        KeyError: If an example is missing required fields.
        ValueError: If the data format is invalid.

    Example:
        >>> ground_truths = load_token_level_dataset("my-rag-eval-token-level")
        >>> for gt in ground_truths:
        ...     print(f"Query: {gt.query.text}")
        ...     for span in gt.relevant_spans:
        ...         print(f"  Doc: {span.doc_id}, chars {span.start}-{span.end}")
    """
    client = get_client()
    examples = client.list_examples(dataset_name=dataset_name)

    ground_truths: list[TokenLevelGroundTruth] = []

    for example in examples:
        inputs = example.inputs
        outputs = example.outputs

        if inputs is None:
            raise ValueError(f"Example {example.id} has no inputs")
        if outputs is None:
            raise ValueError(f"Example {example.id} has no outputs")

        # Extract query from inputs
        query_text = inputs.get("query")
        if query_text is None:
            raise KeyError(f"Example {example.id} is missing 'query' in inputs")

        # Extract spans from outputs
        spans_data = outputs.get("relevant_spans")
        if spans_data is None:
            raise KeyError(f"Example {example.id} is missing 'relevant_spans' in outputs")

        # Convert span dictionaries to CharacterSpan objects
        spans: list[CharacterSpan] = []
        for span_dict in spans_data:
            try:
                span = CharacterSpan(
                    doc_id=DocumentId(str(span_dict["doc_id"])),
                    start=int(span_dict["start"]),
                    end=int(span_dict["end"]),
                    text=str(span_dict["text"]),
                )
                spans.append(span)
            except KeyError as e:
                raise KeyError(f"Example {example.id} has span missing required field: {e}") from e
            except (TypeError, ValueError) as e:
                raise ValueError(f"Example {example.id} has invalid span data: {e}") from e

        # Get metadata
        metadata: dict[str, Any] = {}
        if example.metadata is not None:
            metadata = dict(example.metadata)

        # Create Query object
        query_id = str(example.id) if example.id else str(uuid.uuid4())
        query = Query(
            id=QueryId(query_id),
            text=QueryText(str(query_text)),
            metadata=metadata,
        )

        ground_truth = TokenLevelGroundTruth(
            query=query,
            relevant_spans=spans,
        )
        ground_truths.append(ground_truth)

    logger.info(
        "Loaded %d token-level ground truth examples from dataset '%s'",
        len(ground_truths),
        dataset_name,
    )

    return ground_truths


__all__ = [
    "get_client",
    "load_chunk_level_dataset",
    "load_token_level_dataset",
]
