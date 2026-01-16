# Repository Guidelines

## Project Overview
This repository contains the design and planned implementation of a Python-based RAG (Retrieval-Augmented Generation) Evaluation System. The system supports chunk-level and token-level evaluation pipelines; see `brainstorm.md` and `implementation_plan.md` for architecture details.

## Project Structure & Module Organization
Current design artifacts live at the repo root:
- `brainstorm.md`: Architecture notes and design decisions.
- `implementation_plan.md`: Step-by-step build plan.
- `CLAUDE.md`: Development guidance and commands.

Planned code layout once implementation begins:
- `src/rag_evaluation_system/`: Core library modules (types, chunkers, embedders, vector_stores, rerankers, synthetic_datagen, evaluation, langsmith).
- `tests/`: Pytest suites (e.g., `tests/test_metrics.py`).

## Build, Test, and Development Commands
This project uses `uv` for environment management:
- `uv sync --all-extras`: Install dependencies and extras.
- `uv run pytest`: Run the full test suite.
- `uv run pytest tests/test_metrics.py -v`: Run a single test file.
- `uv run pytest --cov=rag_evaluation_system --cov-report=html`: Generate coverage report.
- `uv run ruff check src tests`: Lint code.
- `uv run ruff format src tests`: Format code.
- `uv run ty check src`: Static type checking.

## Coding Style & Naming Conventions
- Python code is formatted with `ruff format`; linted with `ruff check`.
- Type checking uses `ty`.
- Follow descriptive, module-aligned names (e.g., `ChunkerPositionAdapter`, `CharacterSpan`).
- Prefer snake_case for functions/variables and CapWords for classes.

## Testing Guidelines
- Tests use `pytest` with optional coverage via `pytest-cov`.
- Name tests with `test_*.py` and test functions as `test_*`.
- Keep tests close to the feature they validate (e.g., metrics in `tests/test_metrics.py`).

## Commit & Pull Request Guidelines
- No strict commit convention observed; use concise, imperative summaries (e.g., "Add span merging logic").
- PRs should include a clear description, linked issues (if any), and test results or rationale when tests are not run.
- Include examples or screenshots only if relevant (e.g., docs changes).

## Security & Configuration Tips
- Do not commit secrets or API keys.
- If local configuration is needed, use environment variables or ignored files (e.g., `.env`).
