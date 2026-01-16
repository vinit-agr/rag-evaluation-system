"""Primitive type aliases providing semantic meaning beyond bare strings."""
from typing import Literal, NewType

DocumentId = NewType("DocumentId", str)
QueryId = NewType("QueryId", str)
QueryText = NewType("QueryText", str)
ChunkId = NewType("ChunkId", str)
PositionAwareChunkId = NewType("PositionAwareChunkId", str)

EvaluationType = Literal["chunk-level", "token-level"]
