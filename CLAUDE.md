# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **RAG (Retrieval-Augmented Generation) Evaluation Framework** in Python. Currently in the architectural design phase with the complete design documented in `brainstorm.md` but no implementation code yet.

The framework supports two distinct evaluation paradigms:
- **Chunk-Level**: Ground truth is a list of chunk IDs. Simpler but ties evaluation to a specific chunking strategy.
- **Token-Level (Recommended)**: Ground truth is character spans (doc_id, start, end, text). Chunker-independent, enabling fair comparison across different chunking strategies.

## Architecture

### Core Design Principle
Evaluation type is a foundational choice that shapes the entire pipeline:
1. Different LangSmith dataset schemas
2. Different synthetic data generation strategies
3. Different chunker interfaces (adapter pattern for position awareness)
4. Different metric implementations
5. Strong typing that makes incompatible combinations impossible

### Key Types
- `Document`/`Corpus`: Source documents (typically markdown files)
- `Chunk`: Standard chunk without position tracking (for chunk-level)
- `PositionAwareChunk`: Chunk with character positions (for token-level)
- `CharacterSpan`: Character-level ground truth (doc_id, start, end, text)
- `ChunkId`/`PositionAwareChunkId`: Content-hashed IDs with prefixes (`chunk_` or `pa_chunk_`)

### Two Evaluation Pipelines
```
Chunk-Level:                          Token-Level:
Corpus → Chunker → Chunk IDs          Corpus → CharacterSpan extraction
       ↓                                      ↓
ChunkLevelDataGenerator               TokenLevelDataGenerator (NO chunker)
       ↓                                      ↓
LangSmith (chunk IDs)                 LangSmith (character spans)
       ↓                                      ↓
ChunkLevelEvaluation                  TokenLevelEvaluation
(ChunkRecall, ChunkPrecision)         (SpanRecall, SpanPrecision, SpanIoU)
```

### Chunker Adapter Pattern
- `Chunker`: Simple interface returning `List[str]`
- `PositionAwareChunker`: Full interface returning `List[PositionAwareChunk]`
- `ChunkerPositionAdapter`: Wraps any `Chunker` to make it position-aware by finding chunk positions in source text

### Span Merging for Metrics
Overlapping retrieved spans are merged before metric calculation. Each character is counted at most once to prevent sliding-window chunkers from inflating metrics.

## Planned Directory Structure

```
evaluation/
└── metrics/
    ├── base.py
    ├── chunk_level/     # ChunkRecall, ChunkPrecision, ChunkF1
    └── token_level/     # SpanRecall, SpanPrecision, SpanIoU

synthetic_datagen/
├── base.py
├── chunk_level/         # ChunkLevelDataGenerator (requires chunker)
└── token_level/         # TokenLevelDataGenerator (no chunker needed)
```

## Implementation Roadmap

From `brainstorm.md`:
1. Define final type definitions in `types.py`
2. Implement `PositionAwareChunker` interface and adapter
3. Implement `TokenLevelDataGenerator` with excerpt extraction
4. Implement `ChunkLevelDataGenerator` with citation-style query generation
5. Implement span-based metrics with interval merging
6. Implement chunk-based metrics
7. Implement `TokenLevelEvaluation.run()`
8. Implement `ChunkLevelEvaluation.run()`
9. Update VectorStore interface for position metadata
10. Write comprehensive tests
