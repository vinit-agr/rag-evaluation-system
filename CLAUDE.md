# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **RAG (Retrieval-Augmented Generation) Evaluation System** in Python. Currently in the architectural design phase—see `brainstorm.md` for the complete design and `implementation_plan.md` for detailed implementation steps.

The system supports two evaluation paradigms:
- **Chunk-Level**: Ground truth is a list of chunk IDs. Simpler but ties evaluation to a specific chunking strategy.
- **Token-Level (Recommended)**: Ground truth is character spans (doc_id, start, end, text). Chunker-independent, enabling fair comparison across different chunking strategies.

## Build & Development Commands

This project uses **uv** for package management, **ruff** for linting/formatting, and **ty** for type checking.

```bash
# Install dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Run single test file
uv run pytest tests/test_metrics.py -v

# Run with coverage
uv run pytest --cov=rag_evaluation_system --cov-report=html

# Lint and format
uv run ruff check src tests
uv run ruff format src tests

# Type check
uv run ty check src
```

## Architecture

### Core Design Principle

Evaluation type is a foundational choice that shapes the entire pipeline:
1. Different LangSmith dataset schemas
2. Different synthetic data generation strategies
3. Different chunker interfaces (adapter pattern for position awareness)
4. Different metric implementations
5. Strong typing that makes incompatible combinations impossible

### Two Evaluation Pipelines

```
Chunk-Level:                                    Token-Level:
Corpus → Chunker → Chunk IDs                    Corpus → CharacterSpan extraction
       ↓                                               ↓
ChunkLevelSyntheticDatasetGenerator             TokenLevelSyntheticDatasetGenerator (NO chunker)
       ↓                                               ↓
LangSmith (chunk IDs)                           LangSmith (character spans)
       ↓                                               ↓
ChunkLevelEvaluation                            TokenLevelEvaluation
(ChunkRecall, ChunkPrecision, ChunkF1)          (SpanRecall, SpanPrecision, SpanIoU)
```

### Key Types

| Type | Description |
|------|-------------|
| `Document`/`Corpus` | Source documents (typically markdown files) |
| `Chunk` | Standard chunk without position tracking (chunk-level) |
| `PositionAwareChunk` | Chunk with character positions (token-level) |
| `CharacterSpan` | Character-level ground truth (doc_id, start, end, text) |
| `ChunkId` | Content-hashed ID with `chunk_` prefix |
| `PositionAwareChunkId` | Content-hashed ID with `pa_chunk_` prefix |

### Chunker Adapter Pattern

- `Chunker`: Simple interface returning `List[str]`
- `PositionAwareChunker`: Full interface returning `List[PositionAwareChunk]`
- `ChunkerPositionAdapter`: Wraps any `Chunker` to make it position-aware by finding chunk positions in source text

### Span Merging for Metrics

Overlapping retrieved spans are merged before metric calculation. Each character is counted at most once to prevent sliding-window chunkers from inflating metrics.

## Project Structure

```
src/rag_evaluation_system/
├── types/                  # Pydantic models and type definitions
├── chunkers/               # Chunker interfaces and implementations
├── embedders/              # Embedding interfaces (OpenAI, SentenceTransformers)
├── vector_stores/          # Vector store interfaces (ChromaDB)
├── rerankers/              # Reranker interfaces (Cohere)
├── synthetic_datagen/      # Synthetic data generators (chunk-level, token-level)
├── evaluation/             # Evaluation orchestrators and metrics
└── langsmith/              # LangSmith integration
```

## Implementation Roadmap

See `implementation_plan.md` for detailed steps. High-level phases:
1. Core types with Pydantic models
2. Chunker interfaces and adapter
3. Synthetic data generators
4. Metrics implementation (span merging)
5. Evaluation orchestrators
6. LangSmith integration
