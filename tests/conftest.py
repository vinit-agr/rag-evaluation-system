"""Shared test fixtures."""
import pytest

from rag_evaluation_system.types import (
    CharacterSpan,
    Corpus,
    Document,
    DocumentId,
)


@pytest.fixture
def sample_document() -> Document:
    return Document(
        id=DocumentId("test_doc.md"),
        content=(
            "This is a test document. It has multiple sentences. "
            "Each sentence can be a chunk."
        ),
    )


@pytest.fixture
def sample_corpus(sample_document: Document) -> Corpus:
    return Corpus(documents=[sample_document])


@pytest.fixture
def sample_spans() -> list[CharacterSpan]:
    return [
        CharacterSpan(doc_id=DocumentId("doc1"), start=0, end=50, text="x" * 50),
        CharacterSpan(doc_id=DocumentId("doc1"), start=30, end=80, text="x" * 50),
        CharacterSpan(doc_id=DocumentId("doc2"), start=0, end=100, text="x" * 100),
    ]


@pytest.fixture
def mock_embedder():
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
