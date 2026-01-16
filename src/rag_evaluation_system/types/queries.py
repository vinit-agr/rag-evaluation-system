"""Query types for retrieval evaluation."""
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .primitives import QueryId, QueryText


class Query(BaseModel):
    """A query/question for retrieval evaluation."""

    model_config = ConfigDict(frozen=True)

    id: QueryId
    text: QueryText
    metadata: dict[str, Any] = Field(default_factory=dict)
