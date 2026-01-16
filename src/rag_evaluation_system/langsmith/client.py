"""LangSmith client utilities."""
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


def get_client() -> Client:
    """Get LangSmith client (uses LANGCHAIN_API_KEY env var)."""
    return Client()


def load_chunk_level_dataset(dataset_name: str) -> list[ChunkLevelGroundTruth]:
    """Load chunk-level ground truth from LangSmith."""
    client = get_client()
    examples = list(client.list_examples(dataset_name=dataset_name))

    ground_truth: list[ChunkLevelGroundTruth] = []

    for i, example in enumerate(examples):
        query_text = example.inputs.get("query", "")
        chunk_ids = example.outputs.get("relevant_chunk_ids", [])

        ground_truth.append(
            ChunkLevelGroundTruth(
                query=Query(id=QueryId(f"q_{i}"), text=QueryText(query_text)),
                relevant_chunk_ids=[ChunkId(cid) for cid in chunk_ids],
            )
        )

    return ground_truth


def load_token_level_dataset(dataset_name: str) -> list[TokenLevelGroundTruth]:
    """Load token-level ground truth from LangSmith."""
    client = get_client()
    examples = list(client.list_examples(dataset_name=dataset_name))

    ground_truth: list[TokenLevelGroundTruth] = []

    for i, example in enumerate(examples):
        query_text = example.inputs.get("query", "")
        spans_data = example.outputs.get("relevant_spans", [])

        spans = [
            CharacterSpan(
                doc_id=DocumentId(span["doc_id"]),
                start=span["start"],
                end=span["end"],
                text=span["text"],
            )
            for span in spans_data
        ]

        ground_truth.append(
            TokenLevelGroundTruth(
                query=Query(id=QueryId(f"q_{i}"), text=QueryText(query_text)),
                relevant_spans=spans,
            )
        )

    return ground_truth
