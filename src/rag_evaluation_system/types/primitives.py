"""Primitive type aliases for the RAG evaluation system.

These type aliases provide semantic meaning and type safety beyond bare strings.
Using these instead of `str` makes the code self-documenting and helps catch
type mismatches at development time.
"""

from typing import Literal, NewType

# Unique identifier for a document in the corpus.
# Format: typically the filename or a hash of the file path.
# Example: "rag_overview.md", "doc_a1b2c3d4"
DocumentId = NewType("DocumentId", str)

# Unique identifier for a query/question.
# Format: typically a UUID or hash of the query text.
# Example: "query_f47ac10b"
QueryId = NewType("QueryId", str)

# The actual query/question text that will be used for retrieval.
# Example: "What are the benefits of RAG?"
QueryText = NewType("QueryText", str)

# Unique identifier for a standard chunk (without position tracking).
# Format: "chunk_" prefix + first 12 chars of SHA256 hash of content.
# Example: "chunk_a3f2b1c8d9e0"
# The prefix makes it easy to identify this as a chunk ID at a glance.
ChunkId = NewType("ChunkId", str)

# Unique identifier for a position-aware chunk (with character span tracking).
# Format: "pa_chunk_" prefix + first 12 chars of SHA256 hash of content.
# Example: "pa_chunk_7d9e4f2a1b3c"
# The "pa_" prefix distinguishes these from regular chunk IDs, making it
# immediately clear when you're working with position-aware data.
PositionAwareChunkId = NewType("PositionAwareChunkId", str)

# The type of evaluation to perform. This is a foundational choice that
# determines the shape of ground truth data, metrics used, and chunker requirements.
EvaluationType = Literal["chunk-level", "token-level"]

__all__ = [
    "ChunkId",
    "DocumentId",
    "EvaluationType",
    "PositionAwareChunkId",
    "QueryId",
    "QueryText",
]
