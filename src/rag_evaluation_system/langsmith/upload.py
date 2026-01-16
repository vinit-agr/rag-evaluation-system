"""Utilities for uploading datasets to LangSmith."""
import logging

from langsmith import Client

from rag_evaluation_system.types import ChunkLevelGroundTruth, TokenLevelGroundTruth

logger = logging.getLogger(__name__)


def upload_chunk_level_dataset(
    ground_truth: list[ChunkLevelGroundTruth],
    dataset_name: str | None = None,
) -> str:
    client = Client()
    name = dataset_name or "rag-eval-chunk-level"

    dataset = client.create_dataset(
        dataset_name=name,
        description="Chunk-level RAG evaluation ground truth",
    )

    for gt in ground_truth:
        client.create_example(
            inputs={"query": gt.query.text},
            outputs={"relevant_chunk_ids": [str(cid) for cid in gt.relevant_chunk_ids]},
            metadata=gt.query.metadata,
            dataset_id=dataset.id,
        )

    logger.info("Uploaded %s examples to %s", len(ground_truth), name)
    return name


def upload_token_level_dataset(
    ground_truth: list[TokenLevelGroundTruth],
    dataset_name: str | None = None,
) -> str:
    client = Client()
    name = dataset_name or "rag-eval-token-level"

    dataset = client.create_dataset(
        dataset_name=name,
        description="Token-level RAG evaluation ground truth (character spans)",
    )

    for gt in ground_truth:
        client.create_example(
            inputs={"query": gt.query.text},
            outputs={
                "relevant_spans": [
                    {
                        "doc_id": str(span.doc_id),
                        "start": span.start,
                        "end": span.end,
                        "text": span.text,
                    }
                    for span in gt.relevant_spans
                ]
            },
            metadata=gt.query.metadata,
            dataset_id=dataset.id,
        )

    logger.info("Uploaded %s examples to %s", len(ground_truth), name)
    return name
