"""Query model for the RAG evaluation system.

This module defines the data structure for representing queries/questions
used in retrieval evaluation.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rag_evaluation_system.types.primitives import QueryId, QueryText


class Query(BaseModel):
    """A query/question for retrieval evaluation.

    Represents a single question that will be used to test the retrieval
    pipeline. Contains both the query text and optional metadata.

    Attributes:
        id: Unique identifier for this query.
        text: The actual question text.
        metadata: Arbitrary key-value pairs for additional query information.
            Examples: {"source_doc": "overview.md", "difficulty": "hard"}
    """

    model_config = ConfigDict(frozen=True)

    id: QueryId
    text: QueryText
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "Query",
]
