# RAG Evaluation System - Implementation Plan

This document provides a comprehensive, step-by-step implementation plan for the RAG Evaluation System. It uses modern Python tooling: **uv** (package management), **ty** (type checking), **ruff** (linting/formatting), and **pydantic** (runtime type validation).

---

## Table of Contents

1. [Project Setup](#1-project-setup)
2. [Core Types Module](#2-core-types-module)
3. [Chunker Interfaces](#3-chunker-interfaces)
4. [Embedder & Vector Store Interfaces](#4-embedder--vector-store-interfaces)
5. [Reranker Interface](#5-reranker-interface)
6. [Synthetic Data Generation](#6-synthetic-data-generation)
7. [Metrics Implementation](#7-metrics-implementation)
8. [Evaluation Classes](#8-evaluation-classes)
9. [LangSmith Integration](#9-langsmith-integration)
10. [Built-in Implementations](#10-built-in-implementations)
11. [Testing Strategy](#11-testing-strategy)
12. [Package Publishing](#12-package-publishing)

---

## 1. Project Setup

### 1.1 Initialize Project with uv

```bash
# Create and initialize project
uv init rag-evaluation-system
cd rag-evaluation-system

# Set Python version
uv python pin 3.11
```

### 1.2 Project Structure

Create the following directory structure:

```
rag-evaluation-system/
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── LICENSE
├── .gitignore
├── .python-version
│
├── src/
│   └── rag_evaluation_system/
│       ├── __init__.py
│       ├── py.typed                    # PEP 561 marker for typed package
│       │
│       ├── types/
│       │   ├── __init__.py
│       │   ├── primitives.py           # NewType aliases (DocumentId, QueryId, etc.)
│       │   ├── documents.py            # Document, Corpus
│       │   ├── chunks.py               # Chunk, PositionAwareChunk, CharacterSpan
│       │   ├── queries.py              # Query, QueryText
│       │   ├── ground_truth.py         # ChunkLevelGroundTruth, TokenLevelGroundTruth
│       │   └── results.py              # EvaluationResult, RunOutput types
│       │
│       ├── chunkers/
│       │   ├── __init__.py
│       │   ├── base.py                 # Chunker, PositionAwareChunker ABCs
│       │   ├── adapter.py              # ChunkerPositionAdapter
│       │   ├── recursive_character.py  # RecursiveCharacterChunker
│       │   ├── fixed_token.py          # FixedTokenChunker
│       │   └── semantic.py             # SemanticChunker
│       │
│       ├── embedders/
│       │   ├── __init__.py
│       │   ├── base.py                 # Embedder ABC
│       │   ├── openai.py               # OpenAIEmbedder
│       │   └── sentence_transformers.py # SentenceTransformerEmbedder
│       │
│       ├── vector_stores/
│       │   ├── __init__.py
│       │   ├── base.py                 # VectorStore ABC
│       │   └── chroma.py               # ChromaVectorStore
│       │
│       ├── rerankers/
│       │   ├── __init__.py
│       │   ├── base.py                 # Reranker ABC
│       │   └── cohere.py               # CohereReranker
│       │
│       ├── synthetic_datagen/
│       │   ├── __init__.py
│       │   ├── base.py                 # SyntheticDatasetGenerator ABC
│       │   ├── chunk_level/
│       │   │   ├── __init__.py
│       │   │   └── generator.py        # ChunkLevelSyntheticDatasetGenerator
│       │   └── token_level/
│       │       ├── __init__.py
│       │       └── generator.py        # TokenLevelSyntheticDatasetGenerator
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── chunk_level.py          # ChunkLevelEvaluation
│       │   ├── token_level.py          # TokenLevelEvaluation
│       │   └── metrics/
│       │       ├── __init__.py
│       │       ├── base.py             # ChunkLevelMetric, TokenLevelMetric ABCs
│       │       ├── chunk_level/
│       │       │   ├── __init__.py
│       │       │   ├── recall.py       # ChunkRecall
│       │       │   ├── precision.py    # ChunkPrecision
│       │       │   └── f1.py           # ChunkF1
│       │       └── token_level/
│       │           ├── __init__.py
│       │           ├── recall.py       # SpanRecall
│       │           ├── precision.py    # SpanPrecision
│       │           ├── iou.py          # SpanIoU
│       │           └── utils.py        # Span merging utilities
│       │
│       ├── langsmith/
│       │   ├── __init__.py
│       │   ├── client.py               # LangSmith client wrapper
│       │   ├── schemas.py              # Dataset schemas
│       │   └── upload.py               # Upload utilities
│       │
│       └── utils/
│           ├── __init__.py
│           ├── hashing.py              # Chunk ID generation
│           └── text.py                 # Text processing utilities
│
└── tests/
    ├── __init__.py
    ├── conftest.py                     # Pytest fixtures
    ├── types/
    │   └── test_chunks.py
    ├── chunkers/
    │   ├── test_adapter.py
    │   └── test_recursive_character.py
    ├── evaluation/
    │   └── metrics/
    │       ├── test_chunk_level.py
    │       └── test_token_level.py
    ├── synthetic_datagen/
    │   ├── test_chunk_level_generator.py
    │   └── test_token_level_generator.py
    └── integration/
        ├── test_chunk_level_workflow.py
        └── test_token_level_workflow.py
```

### 1.3 pyproject.toml Configuration

```toml
[project]
name = "rag-evaluation-system"
version = "0.1.0"
description = "A comprehensive framework for evaluating RAG retrieval pipelines"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.11"
authors = [
    { name = "Your Name", email = "you@example.com" }
]
keywords = ["rag", "evaluation", "retrieval", "llm", "langsmith"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Typing :: Typed",
]

dependencies = [
    "pydantic>=2.0",
    "langsmith>=0.1.0",
]

[project.optional-dependencies]
openai = ["openai>=1.0"]
cohere = ["cohere>=5.0"]
chroma = ["chromadb>=0.4"]
sentence-transformers = ["sentence-transformers>=2.0"]
all = [
    "openai>=1.0",
    "cohere>=5.0",
    "chromadb>=0.4",
    "sentence-transformers>=2.0",
]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.4",
    "ty>=0.0.1a0",  # Astral's type checker (or use pyright if ty not available)
    "pre-commit>=3.0",
    "mypy>=1.10",   # Fallback type checker
]

[project.urls]
Homepage = "https://github.com/yourusername/rag-evaluation-system"
Documentation = "https://github.com/yourusername/rag-evaluation-system#readme"
Repository = "https://github.com/yourusername/rag-evaluation-system"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/rag_evaluation_system"]

# =============================================================================
# RUFF CONFIGURATION
# =============================================================================
[tool.ruff]
target-version = "py311"
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # Pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
    "ARG",    # flake8-unused-arguments
    "SIM",    # flake8-simplify
    "TCH",    # flake8-type-checking
    "PTH",    # flake8-use-pathlib
    "RUF",    # Ruff-specific rules
]
ignore = [
    "E501",   # Line too long (handled by formatter)
    "B008",   # Do not perform function calls in argument defaults
]

[tool.ruff.lint.isort]
known-first-party = ["rag_evaluation_system"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
docstring-code-format = true

# =============================================================================
# TYPE CHECKING CONFIGURATION
# =============================================================================
[tool.ty]
# Astral's ty configuration (if available)
python-version = "3.11"

[tool.mypy]
# Fallback mypy configuration
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
module = "chromadb.*"
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "sentence_transformers.*"
ignore_missing_imports = true

# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--tb=short",
    "--strict-markers",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
]

[tool.coverage.run]
source = ["src/rag_evaluation_system"]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
    "if __name__ == .__main__.:",
    "@abstractmethod",
]
```

### 1.4 Install Dependencies

```bash
# Install project with all optional dependencies and dev tools
uv sync --all-extras

# Or install specific extras
uv sync --extra openai --extra chroma --extra dev
```

### 1.5 Configure Development Tools

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: local
    hooks:
      - id: ty
        name: ty type check
        entry: uv run ty check src
        language: system
        types: [python]
        pass_filenames: false
```

Initialize pre-commit:

```bash
uv run pre-commit install
```

---

## 2. Core Types Module

### 2.1 Primitive Types (`types/primitives.py`)

**Purpose**: Define semantic type aliases using `NewType` for compile-time type safety.

**Implementation Details**:

```python
"""Primitive type aliases providing semantic meaning beyond bare strings."""
from typing import NewType

# Document identifier - typically filename or hash
DocumentId = NewType("DocumentId", str)

# Query identifier - UUID or hash
QueryId = NewType("QueryId", str)

# The actual query text
QueryText = NewType("QueryText", str)

# Standard chunk ID: "chunk_" + 12-char SHA256
ChunkId = NewType("ChunkId", str)

# Position-aware chunk ID: "pa_chunk_" + 12-char SHA256
PositionAwareChunkId = NewType("PositionAwareChunkId", str)

# Evaluation type literal
EvaluationType = Literal["chunk-level", "token-level"]
```

**Key Design Decisions**:
- Use `NewType` for zero-runtime-cost type aliases
- Prefixed IDs (`chunk_`, `pa_chunk_`) for visual disambiguation
- 12-character hashes balance uniqueness vs. readability

### 2.2 Document Types (`types/documents.py`)

**Purpose**: Define `Document` and `Corpus` Pydantic models.

**Implementation Details**:

```python
"""Document and Corpus models for the evaluation framework."""
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from .primitives import DocumentId


class Document(BaseModel):
    """A source document from the corpus."""

    model_config = ConfigDict(frozen=True)  # Immutable

    id: DocumentId
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def char_count(self) -> int:
        """Return the character count of the document."""
        return len(self.content)


class Corpus(BaseModel):
    """Collection of documents to evaluate against."""

    model_config = ConfigDict(frozen=True)

    documents: list[Document]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_folder(
        cls,
        folder_path: str | Path,
        glob_pattern: str = "**/*.md",
    ) -> "Corpus":
        """Load documents from a folder.

        Args:
            folder_path: Path to the folder containing documents.
            glob_pattern: Glob pattern for matching files.

        Returns:
            Corpus with all matched documents loaded.
        """
        folder = Path(folder_path)
        documents = []

        for file_path in sorted(folder.glob(glob_pattern)):
            content = file_path.read_text(encoding="utf-8")
            doc_id = DocumentId(file_path.name)  # Use filename as ID
            documents.append(Document(id=doc_id, content=content))

        return cls(documents=documents, metadata={"source_folder": str(folder)})

    def get_document(self, doc_id: DocumentId) -> Document | None:
        """Retrieve a document by ID."""
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None

    def __len__(self) -> int:
        return len(self.documents)
```

**Pydantic Benefits**:
- Automatic validation on construction
- Immutability via `frozen=True`
- Serialization to/from JSON for LangSmith
- Clear schema for API contracts

### 2.3 Chunk Types (`types/chunks.py`)

**Purpose**: Define `Chunk`, `PositionAwareChunk`, and `CharacterSpan` models.

**Implementation Details**:

```python
"""Chunk and span types for retrieval evaluation."""
from typing import Any

from pydantic import BaseModel, Field, ConfigDict, model_validator

from .primitives import DocumentId, ChunkId, PositionAwareChunkId


class CharacterSpan(BaseModel):
    """A span of characters in a source document.

    Represents ground truth for token-level evaluation.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: DocumentId
    start: int = Field(ge=0, description="Start position (inclusive, 0-indexed)")
    end: int = Field(ge=0, description="End position (exclusive)")
    text: str = Field(description="The actual text content")

    @model_validator(mode="after")
    def validate_positions(self) -> "CharacterSpan":
        """Ensure end > start and text length matches."""
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be greater than start ({self.start})")
        expected_length = self.end - self.start
        if len(self.text) != expected_length:
            raise ValueError(
                f"text length ({len(self.text)}) doesn't match span length ({expected_length})"
            )
        return self

    @property
    def length(self) -> int:
        """Return the length of this span in characters."""
        return self.end - self.start

    def overlaps(self, other: "CharacterSpan") -> bool:
        """Check if this span overlaps with another."""
        if self.doc_id != other.doc_id:
            return False
        return self.start < other.end and other.start < self.end

    def overlap_chars(self, other: "CharacterSpan") -> int:
        """Calculate the number of overlapping characters."""
        if not self.overlaps(other):
            return 0
        return min(self.end, other.end) - max(self.start, other.start)

    def merge(self, other: "CharacterSpan") -> "CharacterSpan":
        """Merge with an overlapping span. Raises if no overlap or different docs."""
        if not self.overlaps(other):
            raise ValueError("Cannot merge non-overlapping spans")
        # Note: merged text must be reconstructed from source document
        # This method returns a span with placeholder text
        return CharacterSpan(
            doc_id=self.doc_id,
            start=min(self.start, other.start),
            end=max(self.end, other.end),
            text="",  # Must be filled from source
        )


class Chunk(BaseModel):
    """A text chunk without position tracking.

    Used for chunk-level evaluation where exact positions aren't needed.
    """

    model_config = ConfigDict(frozen=True)

    id: ChunkId
    content: str
    doc_id: DocumentId
    metadata: dict[str, Any] = Field(default_factory=dict)


class PositionAwareChunk(BaseModel):
    """A chunk that knows its exact position in the source document.

    Required for token-level evaluation to compute character overlap.
    """

    model_config = ConfigDict(frozen=True)

    id: PositionAwareChunkId
    content: str
    doc_id: DocumentId
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_positions(self) -> "PositionAwareChunk":
        """Ensure end > start and content length matches."""
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be greater than start ({self.start})")
        expected_length = self.end - self.start
        if len(self.content) != expected_length:
            raise ValueError(
                f"content length ({len(self.content)}) doesn't match span ({expected_length})"
            )
        return self

    def to_span(self) -> CharacterSpan:
        """Convert to CharacterSpan for metric calculation."""
        return CharacterSpan(
            doc_id=self.doc_id,
            start=self.start,
            end=self.end,
            text=self.content,
        )
```

### 2.4 Query Types (`types/queries.py`)

```python
"""Query types for retrieval evaluation."""
from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from .primitives import QueryId, QueryText


class Query(BaseModel):
    """A query/question for retrieval evaluation."""

    model_config = ConfigDict(frozen=True)

    id: QueryId
    text: QueryText
    metadata: dict[str, Any] = Field(default_factory=dict)
```

### 2.5 Ground Truth Types (`types/ground_truth.py`)

```python
"""Ground truth types for chunk-level and token-level evaluation."""
from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict

from .primitives import ChunkId, QueryText
from .queries import Query
from .chunks import CharacterSpan


class ChunkLevelGroundTruth(BaseModel):
    """Ground truth for chunk-level evaluation."""

    model_config = ConfigDict(frozen=True)

    query: Query
    relevant_chunk_ids: list[ChunkId]


class TokenLevelGroundTruth(BaseModel):
    """Ground truth for token-level evaluation.

    Note: Ground truth spans are chunker-independent.
    """

    model_config = ConfigDict(frozen=True)

    query: Query
    relevant_spans: list[CharacterSpan]


# =============================================================================
# LANGSMITH DATASET SCHEMAS
# =============================================================================

class ChunkLevelDatasetExample(TypedDict):
    """LangSmith dataset example schema for chunk-level evaluation.

    Follows LangSmith's inputs/outputs/metadata convention.
    """
    inputs: dict[str, QueryText]        # {"query": "What is RAG?"}
    outputs: dict[str, list[ChunkId]]   # {"relevant_chunk_ids": ["chunk_xxx", ...]}
    metadata: dict[str, Any]            # Top-level metadata (source_docs, generation_model, etc.)


class TokenLevelDatasetExample(TypedDict):
    """LangSmith dataset example schema for token-level evaluation.

    Stores full character span data including text for convenience.
    """
    inputs: dict[str, QueryText]              # {"query": "What is RAG?"}
    outputs: dict[str, list[CharacterSpan]]   # {"relevant_spans": [CharacterSpan(...), ...]}
    metadata: dict[str, Any]                  # Top-level metadata (generation_model, etc.)
```

### 2.6 Result Types (`types/results.py`)

```python
"""Result types for evaluation runs."""
from typing import Any

from pydantic import BaseModel, Field

from .primitives import ChunkId
from .chunks import CharacterSpan


class EvaluationResult(BaseModel):
    """Results from an evaluation run."""

    metrics: dict[str, float]
    experiment_url: str | None = None
    raw_results: Any = None


class ChunkLevelRunOutput(BaseModel):
    """Output from retrieval pipeline for chunk-level evaluation."""

    retrieved_chunk_ids: list[ChunkId]


class TokenLevelRunOutput(BaseModel):
    """Output from retrieval pipeline for token-level evaluation."""

    retrieved_spans: list[CharacterSpan]
```

### 2.7 Module Exports (`types/__init__.py`)

```python
"""Type definitions for the RAG evaluation system."""
from .primitives import (
    DocumentId,
    QueryId,
    QueryText,
    ChunkId,
    PositionAwareChunkId,
    EvaluationType,
)
from .documents import Document, Corpus
from .chunks import CharacterSpan, Chunk, PositionAwareChunk
from .queries import Query
from .ground_truth import (
    ChunkLevelGroundTruth,
    TokenLevelGroundTruth,
    ChunkLevelDatasetExample,
    TokenLevelDatasetExample,
)
from .results import EvaluationResult, ChunkLevelRunOutput, TokenLevelRunOutput

__all__ = [
    # Primitives
    "DocumentId",
    "QueryId",
    "QueryText",
    "ChunkId",
    "PositionAwareChunkId",
    "EvaluationType",
    # Documents
    "Document",
    "Corpus",
    # Chunks
    "CharacterSpan",
    "Chunk",
    "PositionAwareChunk",
    # Queries
    "Query",
    # Ground Truth
    "ChunkLevelGroundTruth",
    "TokenLevelGroundTruth",
    # LangSmith Dataset Schemas
    "ChunkLevelDatasetExample",
    "TokenLevelDatasetExample",
    # Results
    "EvaluationResult",
    "ChunkLevelRunOutput",
    "TokenLevelRunOutput",
]
```

---

## 3. Chunker Interfaces

### 3.1 Base Chunker Classes (`chunkers/base.py`)

**Purpose**: Define abstract base classes for chunkers.

**Implementation Details**:

```python
"""Abstract base classes for chunkers."""
from abc import ABC, abstractmethod

from rag_evaluation_system.types import Document, PositionAwareChunk


class Chunker(ABC):
    """Base chunker interface - returns text chunks without position tracking.

    Use for chunk-level evaluation or when positions aren't needed.
    """

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Split text into chunks.

        Args:
            text: The full text to chunk.

        Returns:
            List of chunk text strings.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a descriptive name for this chunker configuration."""
        ...


class PositionAwareChunker(ABC):
    """Chunker that tracks character positions in the source document.

    Required for token-level evaluation.
    """

    @abstractmethod
    def chunk_with_positions(self, doc: Document) -> list[PositionAwareChunk]:
        """Split document into position-aware chunks.

        Args:
            doc: The document to chunk.

        Returns:
            List of PositionAwareChunk objects with character positions.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a descriptive name for this chunker configuration."""
        ...
```

### 3.2 Position Adapter (`chunkers/adapter.py`)

**Purpose**: Wrap any `Chunker` to make it position-aware.

**Implementation Details**:

```python
"""Adapter to make any Chunker position-aware."""
import logging
from typing import TYPE_CHECKING

from rag_evaluation_system.types import Document, PositionAwareChunk, PositionAwareChunkId
from rag_evaluation_system.utils.hashing import generate_pa_chunk_id
from .base import Chunker, PositionAwareChunker

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ChunkerPositionAdapter(PositionAwareChunker):
    """Adapter that wraps a regular Chunker to make it position-aware.

    Works by finding each chunk's text in the source document.

    Limitations:
        - May fail if the chunker normalizes whitespace or modifies text
        - May fail if the chunker reorders or combines content
        - Logs warning and skips chunks that can't be located
    """

    def __init__(self, chunker: Chunker):
        """Initialize adapter with a chunker to wrap.

        Args:
            chunker: The chunker to wrap.
        """
        self._chunker = chunker
        self._skipped_chunks: int = 0

    @property
    def name(self) -> str:
        """Return name including wrapped chunker."""
        return f"PositionAdapter({self._chunker.name})"

    @property
    def skipped_chunks(self) -> int:
        """Return count of chunks skipped due to location failures."""
        return self._skipped_chunks

    def chunk_with_positions(self, doc: Document) -> list[PositionAwareChunk]:
        """Split document into position-aware chunks.

        Finds each chunk's position by searching in the source text.
        """
        chunks = self._chunker.chunk(doc.content)
        result: list[PositionAwareChunk] = []
        current_pos = 0

        for chunk_text in chunks:
            # Find chunk in original text starting from current position
            start = doc.content.find(chunk_text, current_pos)

            if start == -1:
                # Try from beginning (handles non-sequential chunkers)
                start = doc.content.find(chunk_text)

            if start == -1:
                # Chunk text not found - chunker may have modified it
                logger.warning(
                    f"Could not locate chunk in source document '{doc.id}'. "
                    f"Chunk may have been modified by chunker. Skipping. "
                    f"Chunk preview: {chunk_text[:50]}..."
                )
                self._skipped_chunks += 1
                continue

            end = start + len(chunk_text)

            result.append(
                PositionAwareChunk(
                    id=generate_pa_chunk_id(chunk_text),
                    content=chunk_text,
                    doc_id=doc.id,
                    start=start,
                    end=end,
                )
            )
            current_pos = end

        return result
```

### 3.3 Hashing Utilities (`utils/hashing.py`)

```python
"""Utilities for generating chunk IDs."""
import hashlib

from rag_evaluation_system.types import ChunkId, PositionAwareChunkId


def generate_chunk_id(content: str) -> ChunkId:
    """Generate a standard chunk ID from content.

    Format: "chunk_" + first 12 chars of SHA256 hash.
    """
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return ChunkId(f"chunk_{content_hash}")


def generate_pa_chunk_id(content: str) -> PositionAwareChunkId:
    """Generate a position-aware chunk ID from content.

    Format: "pa_chunk_" + first 12 chars of SHA256 hash.
    """
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return PositionAwareChunkId(f"pa_chunk_{content_hash}")
```

### 3.4 RecursiveCharacterChunker (`chunkers/recursive_character.py`)

**Purpose**: Built-in implementation of a recursive character text splitter.

**Implementation Details**:

```python
"""Recursive character text splitter implementation."""
from rag_evaluation_system.types import Document, PositionAwareChunk
from rag_evaluation_system.utils.hashing import generate_pa_chunk_id
from .base import Chunker, PositionAwareChunker


class RecursiveCharacterChunker(Chunker, PositionAwareChunker):
    """Recursive character text splitter.

    Splits text by trying a list of separators in order, keeping chunks
    under the size limit while preserving semantic boundaries.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ):
        """Initialize the chunker.

        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Number of overlapping characters between chunks.
            separators: List of separators to try, in order.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or self.DEFAULT_SEPARATORS

    @property
    def name(self) -> str:
        return f"RecursiveCharacter(size={self._chunk_size}, overlap={self._chunk_overlap})"

    def chunk(self, text: str) -> list[str]:
        """Split text into chunks."""
        return self._split_text(text, self._separators)

    def chunk_with_positions(self, doc: Document) -> list[PositionAwareChunk]:
        """Split document into position-aware chunks."""
        chunks = self._split_text_with_positions(doc.content, self._separators)
        return [
            PositionAwareChunk(
                id=generate_pa_chunk_id(text),
                content=text,
                doc_id=doc.id,
                start=start,
                end=end,
            )
            for text, start, end in chunks
        ]

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using separators."""
        # Implementation: standard recursive character splitting algorithm
        # Returns list of chunk strings
        ...

    def _split_text_with_positions(
        self,
        text: str,
        separators: list[str],
    ) -> list[tuple[str, int, int]]:
        """Split text and track positions.

        Returns:
            List of (chunk_text, start, end) tuples.
        """
        # Implementation: same algorithm but track character positions
        ...
```

---

## 4. Embedder & Vector Store Interfaces

### 4.1 Embedder Base Class (`embedders/base.py`)

```python
"""Abstract base class for embedders."""
from abc import ABC, abstractmethod


class Embedder(ABC):
    """Base class for text embedding models."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query.

        Some models use different embeddings for queries vs documents.

        Args:
            query: Query text to embed.

        Returns:
            Embedding vector.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a descriptive name for this embedder."""
        ...
```

### 4.2 OpenAI Embedder (`embedders/openai.py`)

```python
"""OpenAI embeddings implementation."""
from typing import TYPE_CHECKING

from .base import Embedder

if TYPE_CHECKING:
    from openai import OpenAI


class OpenAIEmbedder(Embedder):
    """OpenAI text embeddings.

    Requires: pip install rag-evaluation-system[openai]
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        client: "OpenAI | None" = None,
    ):
        """Initialize OpenAI embedder.

        Args:
            model: OpenAI embedding model name.
            client: Optional OpenAI client. Creates one if not provided.
        """
        try:
            from openai import OpenAI as OpenAIClient
        except ImportError as e:
            raise ImportError(
                "OpenAI package required. Install with: "
                "pip install rag-evaluation-system[openai]"
            ) from e

        self._model = model
        self._client = client or OpenAIClient()
        self._dimension = self._get_dimension()

    @property
    def name(self) -> str:
        return f"OpenAI({self._model})"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_dimension(self) -> int:
        """Get embedding dimension by making a test call."""
        # Known dimensions for common models
        known_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return known_dims.get(self._model, 1536)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a query."""
        return self.embed([query])[0]
```

### 4.3 Vector Store Base Class (`vector_stores/base.py`)

```python
"""Abstract base class for vector stores."""
from abc import ABC, abstractmethod

from rag_evaluation_system.types import PositionAwareChunk


class VectorStore(ABC):
    """Base class for vector stores.

    Vector stores must preserve position metadata for token-level evaluation.
    """

    @abstractmethod
    def add(
        self,
        chunks: list[PositionAwareChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add chunks with their embeddings.

        Implementation must store position metadata (doc_id, start, end).

        Args:
            chunks: List of position-aware chunks.
            embeddings: Corresponding embedding vectors.
        """
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[PositionAwareChunk]:
        """Search for similar chunks.

        Must return chunks with position info reconstructed from metadata.

        Args:
            query_embedding: Query embedding vector.
            k: Number of results to return.

        Returns:
            List of similar chunks with position information.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all data from the store."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a descriptive name for this vector store."""
        ...
```

### 4.4 Chroma Vector Store (`vector_stores/chroma.py`)

```python
"""ChromaDB vector store implementation."""
from typing import TYPE_CHECKING
import uuid

from rag_evaluation_system.types import PositionAwareChunk, PositionAwareChunkId, DocumentId
from .base import VectorStore

if TYPE_CHECKING:
    import chromadb


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store.

    Requires: pip install rag-evaluation-system[chroma]
    """

    def __init__(
        self,
        collection_name: str | None = None,
        persist_directory: str | None = None,
    ):
        """Initialize Chroma vector store.

        Args:
            collection_name: Name for the collection. Auto-generated if not provided.
            persist_directory: Directory for persistence. In-memory if not provided.
        """
        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "ChromaDB package required. Install with: "
                "pip install rag-evaluation-system[chroma]"
            ) from e

        self._collection_name = collection_name or f"rag_eval_{uuid.uuid4().hex[:8]}"

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def name(self) -> str:
        return f"Chroma({self._collection_name})"

    def add(
        self,
        chunks: list[PositionAwareChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add chunks with position metadata."""
        if not chunks:
            return

        self._collection.add(
            ids=[str(chunk.id) for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.content for chunk in chunks],
            metadatas=[
                {
                    "doc_id": str(chunk.doc_id),
                    "start": chunk.start,
                    "end": chunk.end,
                    **chunk.metadata,
                }
                for chunk in chunks
            ],
        )

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[PositionAwareChunk]:
        """Search and reconstruct chunks with positions."""
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas"],
        )

        chunks: list[PositionAwareChunk] = []

        if not results["ids"][0]:
            return chunks

        for i, chunk_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            content = results["documents"][0][i]

            chunks.append(
                PositionAwareChunk(
                    id=PositionAwareChunkId(chunk_id),
                    content=content,
                    doc_id=DocumentId(metadata["doc_id"]),
                    start=metadata["start"],
                    end=metadata["end"],
                )
            )

        return chunks

    def clear(self) -> None:
        """Clear the collection."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
```

---

## 5. Reranker Interface

### 5.1 Reranker Base Class (`rerankers/base.py`)

```python
"""Abstract base class for rerankers."""
from abc import ABC, abstractmethod

from rag_evaluation_system.types import PositionAwareChunk


class Reranker(ABC):
    """Base class for reranking models."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[PositionAwareChunk],
        top_k: int | None = None,
    ) -> list[PositionAwareChunk]:
        """Rerank chunks based on relevance to query.

        Args:
            query: The query text.
            chunks: Chunks to rerank.
            top_k: Optional limit on returned chunks.

        Returns:
            Reranked list of chunks.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a descriptive name for this reranker."""
        ...
```

### 5.2 Cohere Reranker (`rerankers/cohere.py`)

```python
"""Cohere reranker implementation."""
from typing import TYPE_CHECKING

from rag_evaluation_system.types import PositionAwareChunk
from .base import Reranker

if TYPE_CHECKING:
    import cohere


class CohereReranker(Reranker):
    """Cohere reranking model.

    Requires: pip install rag-evaluation-system[cohere]
    """

    def __init__(
        self,
        model: str = "rerank-english-v3.0",
        client: "cohere.Client | None" = None,
    ):
        """Initialize Cohere reranker.

        Args:
            model: Cohere rerank model name.
            client: Optional Cohere client.
        """
        try:
            import cohere
        except ImportError as e:
            raise ImportError(
                "Cohere package required. Install with: "
                "pip install rag-evaluation-system[cohere]"
            ) from e

        self._model = model
        self._client = client or cohere.Client()

    @property
    def name(self) -> str:
        return f"Cohere({self._model})"

    def rerank(
        self,
        query: str,
        chunks: list[PositionAwareChunk],
        top_k: int | None = None,
    ) -> list[PositionAwareChunk]:
        """Rerank chunks using Cohere."""
        if not chunks:
            return []

        documents = [chunk.content for chunk in chunks]

        response = self._client.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=top_k or len(chunks),
        )

        # Reorder chunks based on rerank results
        reranked: list[PositionAwareChunk] = []
        for result in response.results:
            reranked.append(chunks[result.index])

        return reranked
```

---

## 6. Synthetic Data Generation

### 6.1 Base Generator (`synthetic_datagen/base.py`)

```python
"""Base class for synthetic data generators."""
from abc import ABC, abstractmethod
from typing import Any

from rag_evaluation_system.types import Corpus


class SyntheticDatasetGenerator(ABC):
    """Base class for synthetic data generation."""

    def __init__(self, llm_client: Any, corpus: Corpus):
        """Initialize generator.

        Args:
            llm_client: LLM client (OpenAI, Anthropic, etc.)
            corpus: Document corpus to generate from.
        """
        self._llm = llm_client
        self._corpus = corpus

    @property
    def corpus(self) -> Corpus:
        """Return the corpus."""
        return self._corpus
```

### 6.2 Chunk-Level Generator (`synthetic_datagen/chunk_level/generator.py`)

**Purpose**: Generate synthetic QA pairs with chunk ID ground truth.

**Implementation Details**:

```python
"""Chunk-level synthetic data generator."""
import json
import logging
from typing import Any

from pydantic import BaseModel

from rag_evaluation_system.types import (
    Corpus,
    Query,
    QueryId,
    QueryText,
    ChunkId,
    ChunkLevelGroundTruth,
)
from rag_evaluation_system.chunkers.base import Chunker
from rag_evaluation_system.utils.hashing import generate_chunk_id
from ..base import SyntheticDatasetGenerator

logger = logging.getLogger(__name__)


class GeneratedQAPair(BaseModel):
    """A generated query with relevant chunk IDs."""
    query: str
    relevant_chunk_ids: list[str]


class ChunkLevelSyntheticDatasetGenerator(SyntheticDatasetGenerator):
    """Generate synthetic QA pairs with chunk-level ground truth.

    Requires a chunker because chunk IDs must exist before referencing
    them in ground truth.
    """

    SYSTEM_PROMPT = '''You are an expert at generating evaluation data for RAG systems.
Given chunks from a document with their IDs, generate questions that can be
answered using specific chunks. For each question, list the chunk IDs that
contain the answer.

Output JSON format:
{
    "qa_pairs": [
        {
            "query": "What is...?",
            "relevant_chunk_ids": ["chunk_xxx", "chunk_yyy"]
        }
    ]
}'''

    def __init__(
        self,
        llm_client: Any,
        corpus: Corpus,
        chunker: Chunker,
    ):
        """Initialize generator.

        Args:
            llm_client: LLM client for generating queries.
            corpus: Document corpus.
            chunker: Chunker to use. Ground truth is tied to this chunker.
        """
        super().__init__(llm_client, corpus)
        self._chunker = chunker
        self._chunk_index: dict[ChunkId, str] = {}

    def generate(
        self,
        queries_per_doc: int = 5,
        upload_to_langsmith: bool = True,
        dataset_name: str | None = None,
    ) -> list[ChunkLevelGroundTruth]:
        """Generate synthetic queries with chunk ID ground truth.

        Process:
        1. Chunk all documents and build chunk index
        2. For each document's chunks:
           - Present chunks with IDs to LLM
           - LLM generates queries + chunk citations
        3. Validate chunk IDs exist
        4. Optionally upload to LangSmith

        Args:
            queries_per_doc: Number of queries to generate per document.
            upload_to_langsmith: Whether to upload to LangSmith.
            dataset_name: LangSmith dataset name.

        Returns:
            List of ChunkLevelGroundTruth objects.
        """
        # Step 1: Chunk all documents
        logger.info("Chunking documents...")
        all_chunks = self._build_chunk_index()

        # Step 2: Generate queries per document
        ground_truth: list[ChunkLevelGroundTruth] = []

        for doc in self._corpus.documents:
            doc_chunks = [
                (chunk_id, content)
                for chunk_id, content in all_chunks.items()
                if chunk_id.startswith("chunk_")  # Filter by doc if needed
            ]

            logger.info(f"Generating {queries_per_doc} queries for {doc.id}")
            qa_pairs = self._generate_qa_pairs(doc_chunks, queries_per_doc)

            for qa in qa_pairs:
                # Validate chunk IDs
                valid_ids = [
                    ChunkId(cid) for cid in qa.relevant_chunk_ids
                    if cid in self._chunk_index
                ]

                if not valid_ids:
                    logger.warning(f"No valid chunk IDs for query: {qa.query[:50]}...")
                    continue

                ground_truth.append(
                    ChunkLevelGroundTruth(
                        query=Query(
                            id=QueryId(f"q_{len(ground_truth)}"),
                            text=QueryText(qa.query),
                            metadata={"source_doc": str(doc.id)},
                        ),
                        relevant_chunk_ids=valid_ids,
                    )
                )

        # Step 3: Upload to LangSmith if requested
        if upload_to_langsmith:
            self._upload_to_langsmith(ground_truth, dataset_name)

        return ground_truth

    def _build_chunk_index(self) -> dict[ChunkId, str]:
        """Chunk all documents and build ID -> content index."""
        for doc in self._corpus.documents:
            chunks = self._chunker.chunk(doc.content)
            for chunk_text in chunks:
                chunk_id = generate_chunk_id(chunk_text)
                self._chunk_index[chunk_id] = chunk_text
        return self._chunk_index

    def _generate_qa_pairs(
        self,
        chunks: list[tuple[ChunkId, str]],
        num_queries: int,
    ) -> list[GeneratedQAPair]:
        """Call LLM to generate QA pairs from chunks."""
        # Format chunks for prompt
        chunk_text = "\n".join(
            f"[{chunk_id}]: {content[:500]}..."
            for chunk_id, content in chunks[:20]  # Limit chunks in prompt
        )

        prompt = f"""Here are chunks from a document:

{chunk_text}

Generate {num_queries} diverse questions that can be answered using these chunks.
For each question, list the chunk IDs that contain the answer."""

        # Call LLM (implementation depends on client type)
        response = self._call_llm(prompt)

        # Parse response
        return self._parse_response(response)

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM. Override for different client types."""
        # Default implementation for OpenAI-compatible client
        response = self._llm.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _parse_response(self, response: str) -> list[GeneratedQAPair]:
        """Parse LLM response into QA pairs."""
        data = json.loads(response)
        return [GeneratedQAPair(**qa) for qa in data.get("qa_pairs", [])]

    def _upload_to_langsmith(
        self,
        ground_truth: list[ChunkLevelGroundTruth],
        dataset_name: str | None,
    ) -> None:
        """Upload ground truth to LangSmith."""
        from rag_evaluation_system.langsmith.upload import upload_chunk_level_dataset
        upload_chunk_level_dataset(ground_truth, dataset_name)
```

### 6.3 Token-Level Generator (`synthetic_datagen/token_level/generator.py`)

**Purpose**: Generate synthetic QA pairs with character span ground truth.

**Implementation Details**:

```python
"""Token-level synthetic data generator."""
import json
import logging
from typing import Any

from pydantic import BaseModel

from rag_evaluation_system.types import (
    Corpus,
    Document,
    Query,
    QueryId,
    QueryText,
    CharacterSpan,
    TokenLevelGroundTruth,
)
from ..base import SyntheticDatasetGenerator

logger = logging.getLogger(__name__)


class ExtractedExcerpt(BaseModel):
    """An excerpt extracted by the LLM."""
    text: str


class GeneratedQAWithExcerpts(BaseModel):
    """A generated query with relevant excerpts."""
    query: str
    excerpts: list[str]


class TokenLevelSyntheticDatasetGenerator(SyntheticDatasetGenerator):
    """Generate synthetic QA pairs with character span ground truth.

    NO chunker required. Ground truth is chunker-independent.
    """

    QUERY_GENERATION_PROMPT = '''You are an expert at generating evaluation questions.
Given a document, generate diverse questions that can be answered using
specific passages from the document.

Output JSON format:
{
    "questions": ["What is...?", "How does...?", ...]
}'''

    EXCERPT_EXTRACTION_PROMPT = '''You are an expert at identifying relevant text.
Given a document and a question, extract the exact passages that answer
the question. Copy the text VERBATIM - do not paraphrase or summarize.

Output JSON format:
{
    "excerpts": ["exact text from document...", ...]
}'''

    def __init__(self, llm_client: Any, corpus: Corpus):
        """Initialize generator.

        Args:
            llm_client: LLM client for generating queries.
            corpus: Document corpus.
        """
        super().__init__(llm_client, corpus)

    def generate(
        self,
        queries_per_doc: int = 5,
        upload_to_langsmith: bool = True,
        dataset_name: str | None = None,
    ) -> list[TokenLevelGroundTruth]:
        """Generate synthetic queries with character span ground truth.

        Process:
        1. For each document:
           - Generate queries about the content
           - Extract verbatim excerpts that answer each query
           - Find character positions of excerpts
        2. Create CharacterSpan ground truth
        3. Optionally upload to LangSmith

        Args:
            queries_per_doc: Number of queries per document.
            upload_to_langsmith: Whether to upload to LangSmith.
            dataset_name: LangSmith dataset name.

        Returns:
            List of TokenLevelGroundTruth objects.
        """
        ground_truth: list[TokenLevelGroundTruth] = []
        query_counter = 0

        for doc in self._corpus.documents:
            logger.info(f"Processing document: {doc.id}")

            # Step 1: Generate questions
            questions = self._generate_questions(doc, queries_per_doc)

            for question in questions:
                # Step 2: Extract excerpts for this question
                excerpts = self._extract_excerpts(doc, question)

                # Step 3: Find character positions
                spans = self._find_span_positions(doc, excerpts)

                if not spans:
                    logger.warning(f"No spans found for: {question[:50]}...")
                    continue

                ground_truth.append(
                    TokenLevelGroundTruth(
                        query=Query(
                            id=QueryId(f"q_{query_counter}"),
                            text=QueryText(question),
                            metadata={"source_doc": str(doc.id)},
                        ),
                        relevant_spans=spans,
                    )
                )
                query_counter += 1

        # Step 4: Upload to LangSmith
        if upload_to_langsmith:
            self._upload_to_langsmith(ground_truth, dataset_name)

        return ground_truth

    def _generate_questions(self, doc: Document, num_queries: int) -> list[str]:
        """Generate questions about a document."""
        prompt = f"""Document:
{doc.content[:8000]}

Generate {num_queries} diverse questions about this document."""

        response = self._call_llm(
            self.QUERY_GENERATION_PROMPT,
            prompt,
        )

        data = json.loads(response)
        return data.get("questions", [])

    def _extract_excerpts(self, doc: Document, question: str) -> list[str]:
        """Extract verbatim excerpts that answer a question."""
        prompt = f"""Document:
{doc.content[:8000]}

Question: {question}

Extract the exact passages that answer this question. Copy verbatim."""

        response = self._call_llm(
            self.EXCERPT_EXTRACTION_PROMPT,
            prompt,
        )

        data = json.loads(response)
        return data.get("excerpts", [])

    def _find_span_positions(
        self,
        doc: Document,
        excerpts: list[str],
    ) -> list[CharacterSpan]:
        """Find character positions of excerpts in document."""
        spans: list[CharacterSpan] = []

        for excerpt in excerpts:
            # Try exact match first
            start = doc.content.find(excerpt)

            if start == -1:
                # Try fuzzy matching for minor LLM variations
                start = self._fuzzy_find(doc.content, excerpt)

            if start == -1:
                logger.warning(
                    f"Could not locate excerpt in document {doc.id}: "
                    f"{excerpt[:50]}..."
                )
                continue

            end = start + len(excerpt)

            spans.append(
                CharacterSpan(
                    doc_id=doc.id,
                    start=start,
                    end=end,
                    text=doc.content[start:end],  # Use actual text from doc
                )
            )

        return spans

    def _fuzzy_find(self, text: str, excerpt: str, threshold: float = 0.9) -> int:
        """Attempt fuzzy matching for excerpts with minor variations.

        Returns start position or -1 if not found.
        """
        # Implementation: sliding window with similarity scoring
        # This handles cases where LLM slightly modified whitespace
        ...
        return -1

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM."""
        response = self._llm.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _upload_to_langsmith(
        self,
        ground_truth: list[TokenLevelGroundTruth],
        dataset_name: str | None,
    ) -> None:
        """Upload ground truth to LangSmith."""
        from rag_evaluation_system.langsmith.upload import upload_token_level_dataset
        upload_token_level_dataset(ground_truth, dataset_name)
```

---

## 7. Metrics Implementation

### 7.1 Base Metric Classes (`evaluation/metrics/base.py`)

```python
"""Base classes for evaluation metrics."""
from abc import ABC, abstractmethod

from rag_evaluation_system.types import ChunkId, CharacterSpan


class ChunkLevelMetric(ABC):
    """Base class for chunk-level metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return metric name."""
        ...

    @abstractmethod
    def calculate(
        self,
        retrieved_chunk_ids: list[ChunkId],
        ground_truth_chunk_ids: list[ChunkId],
    ) -> float:
        """Calculate metric value.

        Args:
            retrieved_chunk_ids: IDs of retrieved chunks.
            ground_truth_chunk_ids: IDs of relevant chunks.

        Returns:
            Metric value between 0.0 and 1.0.
        """
        ...


class TokenLevelMetric(ABC):
    """Base class for token-level (character span) metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return metric name."""
        ...

    @abstractmethod
    def calculate(
        self,
        retrieved_spans: list[CharacterSpan],
        ground_truth_spans: list[CharacterSpan],
    ) -> float:
        """Calculate metric value.

        Args:
            retrieved_spans: Spans from retrieved chunks.
            ground_truth_spans: Ground truth spans.

        Returns:
            Metric value between 0.0 and 1.0.
        """
        ...
```

### 7.2 Chunk-Level Metrics (`evaluation/metrics/chunk_level/`)

**recall.py**:

```python
"""Chunk recall metric."""
from rag_evaluation_system.types import ChunkId
from ..base import ChunkLevelMetric


class ChunkRecall(ChunkLevelMetric):
    """What fraction of relevant chunks were retrieved?"""

    @property
    def name(self) -> str:
        return "chunk_recall"

    def calculate(
        self,
        retrieved_chunk_ids: list[ChunkId],
        ground_truth_chunk_ids: list[ChunkId],
    ) -> float:
        if not ground_truth_chunk_ids:
            return 0.0

        retrieved_set = set(retrieved_chunk_ids)
        ground_truth_set = set(ground_truth_chunk_ids)

        intersection = retrieved_set & ground_truth_set
        return len(intersection) / len(ground_truth_set)
```

**precision.py**:

```python
"""Chunk precision metric."""
from rag_evaluation_system.types import ChunkId
from ..base import ChunkLevelMetric


class ChunkPrecision(ChunkLevelMetric):
    """What fraction of retrieved chunks were relevant?"""

    @property
    def name(self) -> str:
        return "chunk_precision"

    def calculate(
        self,
        retrieved_chunk_ids: list[ChunkId],
        ground_truth_chunk_ids: list[ChunkId],
    ) -> float:
        if not retrieved_chunk_ids:
            return 0.0

        retrieved_set = set(retrieved_chunk_ids)
        ground_truth_set = set(ground_truth_chunk_ids)

        intersection = retrieved_set & ground_truth_set
        return len(intersection) / len(retrieved_set)
```

**f1.py**:

```python
"""Chunk F1 metric."""
from rag_evaluation_system.types import ChunkId
from ..base import ChunkLevelMetric
from .recall import ChunkRecall
from .precision import ChunkPrecision


class ChunkF1(ChunkLevelMetric):
    """Harmonic mean of chunk precision and recall."""

    def __init__(self):
        self._recall = ChunkRecall()
        self._precision = ChunkPrecision()

    @property
    def name(self) -> str:
        return "chunk_f1"

    def calculate(
        self,
        retrieved_chunk_ids: list[ChunkId],
        ground_truth_chunk_ids: list[ChunkId],
    ) -> float:
        recall = self._recall.calculate(retrieved_chunk_ids, ground_truth_chunk_ids)
        precision = self._precision.calculate(retrieved_chunk_ids, ground_truth_chunk_ids)

        if recall + precision == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)
```

### 7.3 Token-Level Metrics

**Span Merging Utilities (`evaluation/metrics/token_level/utils.py`)**:

```python
"""Utilities for span operations."""
from rag_evaluation_system.types import CharacterSpan, DocumentId


def merge_overlapping_spans(spans: list[CharacterSpan]) -> list[CharacterSpan]:
    """Merge overlapping spans within each document.

    Uses interval merging algorithm:
    1. Group by document
    2. Sort by start position
    3. Merge overlapping intervals

    Args:
        spans: List of spans (may overlap).

    Returns:
        List of non-overlapping merged spans.
    """
    if not spans:
        return []

    # Group spans by document
    by_doc: dict[DocumentId, list[CharacterSpan]] = {}
    for span in spans:
        by_doc.setdefault(span.doc_id, []).append(span)

    merged: list[CharacterSpan] = []

    for doc_id, doc_spans in by_doc.items():
        # Sort by start position
        sorted_spans = sorted(doc_spans, key=lambda s: s.start)

        # Merge overlapping intervals
        current = sorted_spans[0]

        for span in sorted_spans[1:]:
            if span.start <= current.end:
                # Overlapping - extend current span
                # Note: text must be reconstructed if needed
                current = CharacterSpan(
                    doc_id=current.doc_id,
                    start=current.start,
                    end=max(current.end, span.end),
                    text="",  # Placeholder - reconstruct from source if needed
                )
            else:
                # Non-overlapping - save current and start new
                merged.append(current)
                current = span

        merged.append(current)

    return merged


def calculate_overlap(
    spans_a: list[CharacterSpan],
    spans_b: list[CharacterSpan],
) -> int:
    """Calculate total character overlap between two span lists.

    Each character is counted at most once.

    Args:
        spans_a: First list of spans.
        spans_b: Second list of spans.

    Returns:
        Total overlapping characters.
    """
    # Merge each list first to avoid double-counting
    merged_a = merge_overlapping_spans(spans_a)
    merged_b = merge_overlapping_spans(spans_b)

    total_overlap = 0

    for span_a in merged_a:
        for span_b in merged_b:
            if span_a.doc_id == span_b.doc_id:
                total_overlap += span_a.overlap_chars(span_b)

    return total_overlap
```

**recall.py**:

```python
"""Span recall metric."""
from rag_evaluation_system.types import CharacterSpan
from ..base import TokenLevelMetric
from .utils import merge_overlapping_spans, calculate_overlap


class SpanRecall(TokenLevelMetric):
    """What fraction of ground truth characters were retrieved?

    Measures completeness of retrieval.
    """

    @property
    def name(self) -> str:
        return "span_recall"

    def calculate(
        self,
        retrieved_spans: list[CharacterSpan],
        ground_truth_spans: list[CharacterSpan],
    ) -> float:
        if not ground_truth_spans:
            return 0.0

        # Merge to avoid double-counting
        merged_gt = merge_overlapping_spans(ground_truth_spans)

        total_gt_chars = sum(span.length for span in merged_gt)
        overlap_chars = calculate_overlap(retrieved_spans, ground_truth_spans)

        return min(overlap_chars / total_gt_chars, 1.0)
```

**precision.py**:

```python
"""Span precision metric."""
from rag_evaluation_system.types import CharacterSpan
from ..base import TokenLevelMetric
from .utils import merge_overlapping_spans, calculate_overlap


class SpanPrecision(TokenLevelMetric):
    """What fraction of retrieved characters were relevant?

    Measures efficiency of retrieval.
    """

    @property
    def name(self) -> str:
        return "span_precision"

    def calculate(
        self,
        retrieved_spans: list[CharacterSpan],
        ground_truth_spans: list[CharacterSpan],
    ) -> float:
        if not retrieved_spans:
            return 0.0

        merged_retrieved = merge_overlapping_spans(retrieved_spans)

        total_ret_chars = sum(span.length for span in merged_retrieved)
        overlap_chars = calculate_overlap(retrieved_spans, ground_truth_spans)

        return min(overlap_chars / total_ret_chars, 1.0)
```

**iou.py**:

```python
"""Span IoU (Intersection over Union) metric."""
from rag_evaluation_system.types import CharacterSpan
from ..base import TokenLevelMetric
from .utils import merge_overlapping_spans, calculate_overlap


class SpanIoU(TokenLevelMetric):
    """Intersection over Union of character spans.

    IoU = |intersection| / |union|
    Balances precision and recall in a single metric.
    """

    @property
    def name(self) -> str:
        return "span_iou"

    def calculate(
        self,
        retrieved_spans: list[CharacterSpan],
        ground_truth_spans: list[CharacterSpan],
    ) -> float:
        if not retrieved_spans and not ground_truth_spans:
            return 1.0
        if not retrieved_spans or not ground_truth_spans:
            return 0.0

        merged_retrieved = merge_overlapping_spans(retrieved_spans)
        merged_gt = merge_overlapping_spans(ground_truth_spans)

        intersection = calculate_overlap(retrieved_spans, ground_truth_spans)

        total_retrieved = sum(span.length for span in merged_retrieved)
        total_gt = sum(span.length for span in merged_gt)
        union = total_retrieved + total_gt - intersection

        return intersection / union if union > 0 else 0.0
```

---

## 8. Evaluation Classes

### 8.1 Chunk-Level Evaluation (`evaluation/chunk_level.py`)

```python
"""Chunk-level evaluation orchestrator."""
import logging
from typing import Callable

from rag_evaluation_system.types import (
    Corpus,
    Chunk,
    ChunkId,
    EvaluationResult,
    ChunkLevelGroundTruth,
)
from rag_evaluation_system.chunkers.base import Chunker
from rag_evaluation_system.embedders.base import Embedder
from rag_evaluation_system.vector_stores.base import VectorStore
from rag_evaluation_system.vector_stores.chroma import ChromaVectorStore
from rag_evaluation_system.rerankers.base import Reranker
from rag_evaluation_system.utils.hashing import generate_chunk_id
from .metrics.base import ChunkLevelMetric
from .metrics.chunk_level import ChunkRecall, ChunkPrecision, ChunkF1

logger = logging.getLogger(__name__)


class ChunkLevelEvaluation:
    """Evaluation using chunk-level metrics.

    Compares retrieved chunk IDs against ground truth chunk IDs.
    """

    DEFAULT_METRICS: list[ChunkLevelMetric] = [
        ChunkRecall(),
        ChunkPrecision(),
        ChunkF1(),
    ]

    def __init__(
        self,
        corpus: Corpus,
        langsmith_dataset_name: str,
    ):
        """Initialize evaluation.

        Args:
            corpus: Document corpus to evaluate.
            langsmith_dataset_name: Name of LangSmith dataset with ground truth.
        """
        self._corpus = corpus
        self._dataset_name = langsmith_dataset_name

    def run(
        self,
        chunker: Chunker,
        embedder: Embedder,
        k: int = 5,
        vector_store: VectorStore | None = None,
        reranker: Reranker | None = None,
        metrics: list[ChunkLevelMetric] | None = None,
    ) -> EvaluationResult:
        """Run chunk-level evaluation.

        Pipeline:
        1. Chunk corpus and generate IDs
        2. Embed and index chunks
        3. For each query: retrieve, optionally rerank, compare IDs
        4. Compute metrics

        Args:
            chunker: Chunker to use.
            embedder: Embedder for vectors.
            k: Number of chunks to retrieve.
            vector_store: Optional vector store (defaults to Chroma).
            reranker: Optional reranker.
            metrics: Metrics to compute (defaults to recall, precision, F1).

        Returns:
            EvaluationResult with metrics and experiment URL.
        """
        vector_store = vector_store or ChromaVectorStore()
        metrics = metrics or self.DEFAULT_METRICS

        # Step 1: Chunk and index
        logger.info(f"Chunking corpus with {chunker.name}")
        chunks, chunk_ids = self._chunk_corpus(chunker)

        logger.info(f"Embedding {len(chunks)} chunks with {embedder.name}")
        embeddings = embedder.embed(chunks)

        logger.info(f"Indexing in {vector_store.name}")
        # Convert to PositionAwareChunk for vector store compatibility
        # (positions won't be used for chunk-level metrics)
        pa_chunks = self._to_position_aware(chunks, chunk_ids)
        vector_store.add(pa_chunks, embeddings)

        # Step 2: Load ground truth
        ground_truth = self._load_ground_truth()

        # Step 3: Evaluate
        logger.info(f"Evaluating {len(ground_truth)} queries")
        all_results: dict[str, list[float]] = {m.name: [] for m in metrics}

        for gt in ground_truth:
            # Retrieve
            query_embedding = embedder.embed_query(gt.query.text)
            retrieved_chunks = vector_store.search(query_embedding, k)

            # Optionally rerank
            if reranker:
                retrieved_chunks = reranker.rerank(
                    gt.query.text,
                    retrieved_chunks,
                    top_k=k,
                )

            # Get chunk IDs
            retrieved_ids = [
                ChunkId(str(c.id).replace("pa_chunk_", "chunk_"))
                for c in retrieved_chunks
            ]

            # Calculate metrics
            for metric in metrics:
                score = metric.calculate(retrieved_ids, gt.relevant_chunk_ids)
                all_results[metric.name].append(score)

        # Step 4: Aggregate results
        avg_metrics = {
            name: sum(scores) / len(scores) if scores else 0.0
            for name, scores in all_results.items()
        }

        logger.info(f"Results: {avg_metrics}")

        return EvaluationResult(
            metrics=avg_metrics,
            experiment_url=None,  # Set by LangSmith integration
        )

    def _chunk_corpus(self, chunker: Chunker) -> tuple[list[str], list[ChunkId]]:
        """Chunk all documents and generate IDs."""
        chunks: list[str] = []
        chunk_ids: list[ChunkId] = []

        for doc in self._corpus.documents:
            doc_chunks = chunker.chunk(doc.content)
            for chunk_text in doc_chunks:
                chunks.append(chunk_text)
                chunk_ids.append(generate_chunk_id(chunk_text))

        return chunks, chunk_ids

    def _to_position_aware(
        self,
        chunks: list[str],
        chunk_ids: list[ChunkId],
    ) -> list:
        """Convert to PositionAwareChunk for storage (without real positions)."""
        from rag_evaluation_system.types import PositionAwareChunk, PositionAwareChunkId, DocumentId

        return [
            PositionAwareChunk(
                id=PositionAwareChunkId(str(cid).replace("chunk_", "pa_chunk_")),
                content=text,
                doc_id=DocumentId("unknown"),  # Not needed for chunk-level
                start=0,
                end=len(text),
            )
            for text, cid in zip(chunks, chunk_ids)
        ]

    def _load_ground_truth(self) -> list[ChunkLevelGroundTruth]:
        """Load ground truth from LangSmith."""
        from rag_evaluation_system.langsmith.client import load_chunk_level_dataset
        return load_chunk_level_dataset(self._dataset_name)
```

### 8.2 Token-Level Evaluation (`evaluation/token_level.py`)

```python
"""Token-level evaluation orchestrator."""
import logging
from typing import Union

from rag_evaluation_system.types import (
    Corpus,
    CharacterSpan,
    EvaluationResult,
    TokenLevelGroundTruth,
)
from rag_evaluation_system.chunkers.base import Chunker, PositionAwareChunker
from rag_evaluation_system.chunkers.adapter import ChunkerPositionAdapter
from rag_evaluation_system.embedders.base import Embedder
from rag_evaluation_system.vector_stores.base import VectorStore
from rag_evaluation_system.vector_stores.chroma import ChromaVectorStore
from rag_evaluation_system.rerankers.base import Reranker
from .metrics.base import TokenLevelMetric
from .metrics.token_level import SpanRecall, SpanPrecision, SpanIoU

logger = logging.getLogger(__name__)


class TokenLevelEvaluation:
    """Evaluation using token-level (character span) metrics.

    Compares character overlap between retrieved chunks and ground truth spans.
    """

    DEFAULT_METRICS: list[TokenLevelMetric] = [
        SpanRecall(),
        SpanPrecision(),
        SpanIoU(),
    ]

    def __init__(
        self,
        corpus: Corpus,
        langsmith_dataset_name: str,
    ):
        """Initialize evaluation.

        Args:
            corpus: Document corpus.
            langsmith_dataset_name: LangSmith dataset with ground truth spans.
        """
        self._corpus = corpus
        self._dataset_name = langsmith_dataset_name

    def run(
        self,
        chunker: Union[Chunker, PositionAwareChunker],
        embedder: Embedder,
        k: int = 5,
        vector_store: VectorStore | None = None,
        reranker: Reranker | None = None,
        metrics: list[TokenLevelMetric] | None = None,
    ) -> EvaluationResult:
        """Run token-level evaluation.

        Pipeline:
        1. Chunk corpus with position tracking
        2. Embed and index chunks (with position metadata)
        3. For each query: retrieve, convert to spans, compare
        4. Compute metrics

        Args:
            chunker: Chunker to use (wrapped if not position-aware).
            embedder: Embedder for vectors.
            k: Number of chunks to retrieve.
            vector_store: Optional vector store.
            reranker: Optional reranker.
            metrics: Metrics to compute.

        Returns:
            EvaluationResult with metrics.
        """
        vector_store = vector_store or ChromaVectorStore()
        metrics = metrics or self.DEFAULT_METRICS

        # Ensure position-aware chunker
        if isinstance(chunker, Chunker) and not isinstance(chunker, PositionAwareChunker):
            logger.info(f"Wrapping {chunker.name} with position adapter")
            pa_chunker: PositionAwareChunker = ChunkerPositionAdapter(chunker)
        else:
            pa_chunker = chunker  # type: ignore

        # Step 1: Chunk with positions
        logger.info(f"Chunking corpus with {pa_chunker.name}")
        all_chunks = []
        for doc in self._corpus.documents:
            doc_chunks = pa_chunker.chunk_with_positions(doc)
            all_chunks.extend(doc_chunks)

        logger.info(f"Generated {len(all_chunks)} position-aware chunks")

        # Step 2: Embed and index
        logger.info(f"Embedding chunks with {embedder.name}")
        chunk_texts = [c.content for c in all_chunks]
        embeddings = embedder.embed(chunk_texts)

        logger.info(f"Indexing in {vector_store.name}")
        vector_store.add(all_chunks, embeddings)

        # Step 3: Load ground truth
        ground_truth = self._load_ground_truth()

        # Step 4: Evaluate
        logger.info(f"Evaluating {len(ground_truth)} queries")
        all_results: dict[str, list[float]] = {m.name: [] for m in metrics}

        for gt in ground_truth:
            # Retrieve
            query_embedding = embedder.embed_query(gt.query.text)
            retrieved_chunks = vector_store.search(query_embedding, k)

            # Optionally rerank
            if reranker:
                retrieved_chunks = reranker.rerank(
                    gt.query.text,
                    retrieved_chunks,
                    top_k=k,
                )

            # Convert to spans
            retrieved_spans = [chunk.to_span() for chunk in retrieved_chunks]

            # Calculate metrics
            for metric in metrics:
                score = metric.calculate(retrieved_spans, gt.relevant_spans)
                all_results[metric.name].append(score)

        # Step 5: Aggregate
        avg_metrics = {
            name: sum(scores) / len(scores) if scores else 0.0
            for name, scores in all_results.items()
        }

        logger.info(f"Results: {avg_metrics}")

        return EvaluationResult(metrics=avg_metrics)

    def _load_ground_truth(self) -> list[TokenLevelGroundTruth]:
        """Load ground truth from LangSmith."""
        from rag_evaluation_system.langsmith.client import load_token_level_dataset
        return load_token_level_dataset(self._dataset_name)
```

---

## 9. LangSmith Integration

### 9.1 Client Wrapper (`langsmith/client.py`)

```python
"""LangSmith client utilities."""
from langsmith import Client

from rag_evaluation_system.types import (
    Query,
    QueryId,
    QueryText,
    ChunkId,
    CharacterSpan,
    DocumentId,
    ChunkLevelGroundTruth,
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
                query=Query(
                    id=QueryId(f"q_{i}"),
                    text=QueryText(query_text),
                ),
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
                doc_id=DocumentId(s["doc_id"]),
                start=s["start"],
                end=s["end"],
                text=s["text"],
            )
            for s in spans_data
        ]

        ground_truth.append(
            TokenLevelGroundTruth(
                query=Query(
                    id=QueryId(f"q_{i}"),
                    text=QueryText(query_text),
                ),
                relevant_spans=spans,
            )
        )

    return ground_truth
```

### 9.2 Upload Utilities (`langsmith/upload.py`)

```python
"""Utilities for uploading datasets to LangSmith."""
import logging
from langsmith import Client

from rag_evaluation_system.types import ChunkLevelGroundTruth, TokenLevelGroundTruth

logger = logging.getLogger(__name__)


def upload_chunk_level_dataset(
    ground_truth: list[ChunkLevelGroundTruth],
    dataset_name: str | None = None,
) -> str:
    """Upload chunk-level ground truth to LangSmith.

    Returns:
        Dataset name.
    """
    client = Client()
    name = dataset_name or "rag-eval-chunk-level"

    # Create or get dataset
    dataset = client.create_dataset(
        dataset_name=name,
        description="Chunk-level RAG evaluation ground truth",
    )

    # Create examples
    for gt in ground_truth:
        client.create_example(
            inputs={"query": gt.query.text},
            outputs={
                "relevant_chunk_ids": [str(cid) for cid in gt.relevant_chunk_ids],
            },
            metadata=gt.query.metadata,  # Top-level metadata for LangSmith
            dataset_id=dataset.id,
        )

    logger.info(f"Uploaded {len(ground_truth)} examples to {name}")
    return name


def upload_token_level_dataset(
    ground_truth: list[TokenLevelGroundTruth],
    dataset_name: str | None = None,
) -> str:
    """Upload token-level ground truth to LangSmith.

    Returns:
        Dataset name.
    """
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
                ],
            },
            metadata=gt.query.metadata,  # Top-level metadata for LangSmith
            dataset_id=dataset.id,
        )

    logger.info(f"Uploaded {len(ground_truth)} examples to {name}")
    return name
```

---

## 10. Built-in Implementations

### 10.1 Additional Chunkers

**FixedTokenChunker** (`chunkers/fixed_token.py`):
- Split by token count using tiktoken
- Parameters: `tokens_per_chunk`, `overlap_tokens`

**SemanticChunker** (`chunkers/semantic.py`):
- Split by semantic similarity using embeddings
- Parameters: `embedder`, `similarity_threshold`

### 10.2 Additional Embedders

**SentenceTransformerEmbedder** (`embedders/sentence_transformers.py`):
- Local embedding using sentence-transformers
- Parameters: `model_name`

### 10.3 Package Exports (`__init__.py`)

```python
"""RAG Evaluation System - Comprehensive RAG retrieval evaluation."""
from rag_evaluation_system.types import (
    Document,
    Corpus,
    Chunk,
    PositionAwareChunk,
    CharacterSpan,
    Query,
    ChunkLevelGroundTruth,
    TokenLevelGroundTruth,
    EvaluationResult,
)
from rag_evaluation_system.chunkers import (
    Chunker,
    PositionAwareChunker,
    ChunkerPositionAdapter,
    RecursiveCharacterChunker,
)
from rag_evaluation_system.embedders import Embedder
from rag_evaluation_system.vector_stores import VectorStore
from rag_evaluation_system.rerankers import Reranker
from rag_evaluation_system.synthetic_datagen import (
    ChunkLevelSyntheticDatasetGenerator,
    TokenLevelSyntheticDatasetGenerator,
)
from rag_evaluation_system.evaluation import (
    ChunkLevelEvaluation,
    TokenLevelEvaluation,
)

# Conditional imports for optional dependencies
try:
    from rag_evaluation_system.embedders.openai import OpenAIEmbedder
except ImportError:
    OpenAIEmbedder = None  # type: ignore

try:
    from rag_evaluation_system.vector_stores.chroma import ChromaVectorStore
except ImportError:
    ChromaVectorStore = None  # type: ignore

try:
    from rag_evaluation_system.rerankers.cohere import CohereReranker
except ImportError:
    CohereReranker = None  # type: ignore

__version__ = "0.1.0"

__all__ = [
    # Types
    "Document",
    "Corpus",
    "Chunk",
    "PositionAwareChunk",
    "CharacterSpan",
    "Query",
    "ChunkLevelGroundTruth",
    "TokenLevelGroundTruth",
    "EvaluationResult",
    # Chunkers
    "Chunker",
    "PositionAwareChunker",
    "ChunkerPositionAdapter",
    "RecursiveCharacterChunker",
    # Interfaces
    "Embedder",
    "VectorStore",
    "Reranker",
    # Data Generation
    "ChunkLevelSyntheticDatasetGenerator",
    "TokenLevelSyntheticDatasetGenerator",
    # Evaluation
    "ChunkLevelEvaluation",
    "TokenLevelEvaluation",
    # Implementations (optional)
    "OpenAIEmbedder",
    "ChromaVectorStore",
    "CohereReranker",
]
```

---

## 11. Testing Strategy

### 11.1 Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── unit/
│   ├── types/
│   │   ├── test_character_span.py
│   │   └── test_corpus.py
│   ├── chunkers/
│   │   ├── test_adapter.py
│   │   └── test_recursive_character.py
│   ├── metrics/
│   │   ├── test_chunk_metrics.py
│   │   └── test_span_metrics.py
│   └── utils/
│       └── test_hashing.py
├── integration/
│   ├── test_chunk_level_workflow.py
│   └── test_token_level_workflow.py
└── e2e/
    └── test_full_evaluation.py
```

### 11.2 Key Test Fixtures (`conftest.py`)

```python
"""Shared test fixtures."""
import pytest
from rag_evaluation_system.types import (
    Document,
    Corpus,
    DocumentId,
    CharacterSpan,
    PositionAwareChunk,
    PositionAwareChunkId,
)


@pytest.fixture
def sample_document() -> Document:
    """A sample document for testing."""
    return Document(
        id=DocumentId("test_doc.md"),
        content="This is a test document. It has multiple sentences. Each sentence can be a chunk.",
    )


@pytest.fixture
def sample_corpus(sample_document: Document) -> Corpus:
    """A sample corpus with one document."""
    return Corpus(documents=[sample_document])


@pytest.fixture
def sample_spans() -> list[CharacterSpan]:
    """Sample character spans for metric testing."""
    return [
        CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=50, text="x" * 50),
        CharacterSpan(doc_id=DocumentId("doc1"), start=30, end=80, text="x" * 50),  # Overlapping
        CharacterSpan(doc_id=DocumentId("doc2"), start=0, end=100, text="x" * 100),
    ]


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns fixed-dimension vectors."""
    from rag_evaluation_system.embedders.base import Embedder

    class MockEmbedder(Embedder):
        @property
        def name(self) -> str:
            return "MockEmbedder"

        @property
        def dimension(self) -> int:
            return 128

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.1] * 128 for _ in texts]

        def embed_query(self, query: str) -> list[float]:
            return [0.1] * 128

    return MockEmbedder()
```

### 11.3 Unit Test Examples

**test_character_span.py**:

```python
"""Tests for CharacterSpan."""
import pytest
from rag_evaluation_system.types import CharacterSpan, DocumentId


class TestCharacterSpan:
    def test_length(self):
        span = CharacterSpan(
            doc_id=DocumentId("doc1"),
            start=10,
            end=50,
            text="x" * 40,
        )
        assert span.length == 40

    def test_overlap_same_doc(self):
        span1 = CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=50, text="x" * 50)
        span2 = CharacterSpan(doc_id=DocumentId("doc1"), start=30, end=80, text="x" * 50)

        assert span1.overlaps(span2)
        assert span1.overlap_chars(span2) == 20

    def test_no_overlap_different_docs(self):
        span1 = CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=50, text="x" * 50)
        span2 = CharacterSpan(doc_id=DocumentId("doc2"), start=0, end=50, text="x" * 50)

        assert not span1.overlaps(span2)
        assert span1.overlap_chars(span2) == 0

    def test_validation_end_before_start(self):
        with pytest.raises(ValueError, match="end.*must be greater than start"):
            CharacterSpan(doc_id=DocumentId("doc1"), start=50, end=10, text="x")

    def test_validation_text_length_mismatch(self):
        with pytest.raises(ValueError, match="text length.*doesn't match"):
            CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=50, text="x" * 10)
```

**test_span_metrics.py**:

```python
"""Tests for token-level metrics."""
import pytest
from rag_evaluation_system.types import CharacterSpan, DocumentId
from rag_evaluation_system.evaluation.metrics.token_level import (
    SpanRecall,
    SpanPrecision,
    SpanIoU,
)
from rag_evaluation_system.evaluation.metrics.token_level.utils import merge_overlapping_spans


class TestSpanMerging:
    def test_merge_overlapping(self):
        spans = [
            CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=50, text=""),
            CharacterSpan(doc_id=DocumentId("doc1"), start=30, end=80, text=""),
        ]
        merged = merge_overlapping_spans(spans)

        assert len(merged) == 1
        assert merged[0].start == 0
        assert merged[0].end == 80

    def test_merge_non_overlapping(self):
        spans = [
            CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=50, text=""),
            CharacterSpan(doc_id=DocumentId("doc1"), start=100, end=150, text=""),
        ]
        merged = merge_overlapping_spans(spans)

        assert len(merged) == 2


class TestSpanRecall:
    def test_perfect_recall(self):
        gt = [CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=100, text="x" * 100)]
        retrieved = [CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=100, text="x" * 100)]

        metric = SpanRecall()
        assert metric.calculate(retrieved, gt) == 1.0

    def test_partial_recall(self):
        gt = [CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=100, text="x" * 100)]
        retrieved = [CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=50, text="x" * 50)]

        metric = SpanRecall()
        assert metric.calculate(retrieved, gt) == 0.5

    def test_empty_ground_truth(self):
        metric = SpanRecall()
        assert metric.calculate([], []) == 0.0


class TestSpanPrecision:
    def test_perfect_precision(self):
        gt = [CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=100, text="x" * 100)]
        retrieved = [CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=100, text="x" * 100)]

        metric = SpanPrecision()
        assert metric.calculate(retrieved, gt) == 1.0

    def test_low_precision(self):
        gt = [CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=50, text="x" * 50)]
        retrieved = [CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=100, text="x" * 100)]

        metric = SpanPrecision()
        assert metric.calculate(retrieved, gt) == 0.5


class TestSpanIoU:
    def test_perfect_iou(self):
        gt = [CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=100, text="x" * 100)]
        retrieved = [CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=100, text="x" * 100)]

        metric = SpanIoU()
        assert metric.calculate(retrieved, gt) == 1.0

    def test_partial_overlap_iou(self):
        gt = [CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=100, text="x" * 100)]
        retrieved = [CharacterSpan(doc_id=DocumentId("doc1"), start=50, end=150, text="x" * 100)]

        # Intersection: 50-100 = 50 chars
        # Union: 0-150 = 150 chars
        # IoU = 50/150 = 0.333...
        metric = SpanIoU()
        assert abs(metric.calculate(retrieved, gt) - 0.333) < 0.01
```

### 11.4 Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=rag_evaluation_system --cov-report=html

# Run only unit tests
uv run pytest tests/unit/

# Run only integration tests
uv run pytest tests/integration/ -m integration

# Run specific test file
uv run pytest tests/unit/metrics/test_span_metrics.py -v
```

---

## 12. Package Publishing

### 12.1 Build Package

```bash
# Build source distribution and wheel
uv build
```

### 12.2 Publish to PyPI

```bash
# Publish to TestPyPI first
uv publish --repository testpypi

# Publish to PyPI
uv publish
```

### 12.3 Version Management

Use semantic versioning:
- `0.1.0` - Initial release
- `0.1.1` - Bug fixes
- `0.2.0` - New features (backward compatible)
- `1.0.0` - Stable API

---

## Implementation Checklist

### Phase 1: Project Foundation
- [ ] Initialize project with `uv init`
- [ ] Configure `pyproject.toml` with all settings
- [ ] Set up ruff and ty/mypy configuration
- [ ] Create directory structure
- [ ] Add `py.typed` marker
- [ ] Set up pre-commit hooks

### Phase 2: Core Types
- [ ] Implement `types/primitives.py`
- [ ] Implement `types/documents.py` with Pydantic models
- [ ] Implement `types/chunks.py` with CharacterSpan validation
- [ ] Implement `types/queries.py`
- [ ] Implement `types/ground_truth.py`
- [ ] Implement `types/results.py`
- [ ] Write unit tests for all types

### Phase 3: Chunkers
- [ ] Implement `chunkers/base.py` ABCs
- [ ] Implement `chunkers/adapter.py`
- [ ] Implement `chunkers/recursive_character.py`
- [ ] Implement `utils/hashing.py`
- [ ] Write tests for adapter and recursive chunker

### Phase 4: Embedders & Vector Stores
- [ ] Implement `embedders/base.py`
- [ ] Implement `embedders/openai.py`
- [ ] Implement `vector_stores/base.py`
- [ ] Implement `vector_stores/chroma.py`
- [ ] Write tests with mock embedders

### Phase 5: Rerankers
- [ ] Implement `rerankers/base.py`
- [ ] Implement `rerankers/cohere.py`
- [ ] Write tests

### Phase 6: Metrics
- [ ] Implement `evaluation/metrics/base.py`
- [ ] Implement chunk-level metrics (recall, precision, F1)
- [ ] Implement `evaluation/metrics/token_level/utils.py` (span merging)
- [ ] Implement token-level metrics (recall, precision, IoU)
- [ ] Write comprehensive metric tests

### Phase 7: Synthetic Data Generation
- [ ] Implement `synthetic_datagen/base.py`
- [ ] Implement `synthetic_datagen/chunk_level/generator.py`
- [ ] Implement `synthetic_datagen/token_level/generator.py`
- [ ] Write tests with mocked LLM responses

### Phase 8: LangSmith Integration
- [ ] Implement `langsmith/client.py`
- [ ] Implement `langsmith/upload.py`
- [ ] Implement `langsmith/schemas.py`
- [ ] Write integration tests

### Phase 9: Evaluation Orchestrators
- [ ] Implement `evaluation/chunk_level.py`
- [ ] Implement `evaluation/token_level.py`
- [ ] Write integration tests

### Phase 10: Package & Documentation
- [ ] Create package `__init__.py` with exports
- [ ] Update CLAUDE.md with final structure
- [ ] Write usage examples
- [ ] Build and test package locally
- [ ] Publish to TestPyPI
- [ ] Publish to PyPI

---

## Summary

This implementation plan provides a comprehensive roadmap for building the RAG Evaluation System using modern Python practices:

1. **Modern Tooling**: uv for package management, ruff for linting/formatting, ty for type checking
2. **Type Safety**: Pydantic models with validation, NewType aliases for semantic clarity
3. **Clean Architecture**: Separate ABCs from implementations, optional dependencies
4. **Two Evaluation Paradigms**: Chunk-level (simpler) and Token-level (more granular)
5. **Extensible Design**: Easy to add new chunkers, embedders, vector stores, and metrics

The system enables fair comparison of RAG retrieval pipelines through standardized evaluation against LangSmith-stored ground truth datasets.
