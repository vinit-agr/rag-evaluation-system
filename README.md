# RAG Evaluation System

A comprehensive framework for evaluating RAG (Retrieval-Augmented Generation) retrieval pipelines.

## Overview

This system supports two evaluation paradigms:

- **Chunk-Level**: Ground truth is a list of chunk IDs. Simpler but ties evaluation to a specific chunking strategy.
- **Token-Level (Recommended)**: Ground truth is character spans (doc_id, start, end, text). Chunker-independent, enabling fair comparison across different chunking strategies.

## Installation

```bash
# Install with uv
uv sync

# Install with all optional dependencies
uv sync --all-extras

# Install specific extras
uv sync --extra openai --extra chroma --extra dev
```

## Development

```bash
# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=rag_evaluation_system --cov-report=html

# Lint and format
uv run ruff check src tests
uv run ruff format src tests

# Type check
uv run ty check src
```

## License

MIT
