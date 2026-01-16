"""Tests for Document and Corpus types."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_evaluation_system.types import Corpus, Document, DocumentId


class TestDocument:
    """Tests for the Document model."""

    def test_document_creation_with_valid_data(self) -> None:
        """Test that Document can be created with valid data."""
        doc_id = DocumentId("test_doc.md")
        content = "This is test content."
        metadata = {"author": "Test Author", "version": "1.0"}

        doc = Document(id=doc_id, content=content, metadata=metadata)

        assert doc.id == doc_id
        assert doc.content == content
        assert doc.metadata == metadata

    def test_document_creation_with_minimal_data(self) -> None:
        """Test that Document can be created with just required fields."""
        doc_id = DocumentId("minimal_doc.md")
        content = "Minimal content."

        doc = Document(id=doc_id, content=content)

        assert doc.id == doc_id
        assert doc.content == content
        assert doc.metadata == {}  # Default empty dict

    def test_document_creation_with_empty_content(self) -> None:
        """Test that Document can be created with empty content."""
        doc = Document(id=DocumentId("empty.md"), content="")

        assert doc.content == ""
        assert doc.char_count == 0

    def test_document_char_count_property(self) -> None:
        """Test that char_count property returns correct character count."""
        content = "Hello, World!"
        doc = Document(id=DocumentId("test.md"), content=content)

        assert doc.char_count == len(content)
        assert doc.char_count == 13

    def test_document_char_count_with_unicode(self) -> None:
        """Test char_count with unicode characters."""
        content = "Hello, \u4e16\u754c!"  # Hello, World! in Chinese
        doc = Document(id=DocumentId("unicode.md"), content=content)

        assert doc.char_count == len(content)

    def test_document_char_count_with_multiline(self) -> None:
        """Test char_count with multiline content."""
        content = "Line 1\nLine 2\nLine 3"
        doc = Document(id=DocumentId("multiline.md"), content=content)

        assert doc.char_count == len(content)

    def test_document_immutability(self) -> None:
        """Test that Document is frozen (immutable)."""
        doc = Document(
            id=DocumentId("frozen.md"),
            content="Frozen content",
            metadata={"key": "value"},
        )

        with pytest.raises(ValidationError):  # Pydantic frozen model raises ValidationError
            doc.id = DocumentId("new_id.md")  # type: ignore[misc]

        with pytest.raises(ValidationError):  # Pydantic frozen model raises ValidationError
            doc.content = "New content"  # type: ignore[misc]


class TestCorpus:
    """Tests for the Corpus model."""

    def test_corpus_creation(self) -> None:
        """Test that Corpus can be created with a list of documents."""
        docs = [
            Document(id=DocumentId("doc1.md"), content="Content 1"),
            Document(id=DocumentId("doc2.md"), content="Content 2"),
            Document(id=DocumentId("doc3.md"), content="Content 3"),
        ]
        metadata = {"name": "test_corpus", "version": "1.0"}

        corpus = Corpus(documents=docs, metadata=metadata)

        assert len(corpus.documents) == 3
        assert corpus.metadata == metadata

    def test_corpus_creation_with_empty_documents(self) -> None:
        """Test that Corpus can be created with empty documents list."""
        corpus = Corpus(documents=[])

        assert len(corpus.documents) == 0
        assert len(corpus) == 0

    def test_corpus_from_folder(self, tmp_path: Path) -> None:
        """Test Corpus.from_folder loads markdown files correctly."""
        # Create test markdown files
        (tmp_path / "doc1.md").write_text("# Document 1\n\nContent of doc 1.")
        (tmp_path / "doc2.md").write_text("# Document 2\n\nContent of doc 2.")
        (tmp_path / "doc3.md").write_text("# Document 3\n\nContent of doc 3.")

        corpus = Corpus.from_folder(tmp_path)

        assert len(corpus) == 3
        assert corpus.metadata["source_folder"] == str(tmp_path)
        assert corpus.metadata["glob_pattern"] == "**/*.md"

    def test_corpus_from_folder_with_subdirectories(self, tmp_path: Path) -> None:
        """Test Corpus.from_folder handles subdirectories correctly."""
        # Create nested structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        (tmp_path / "root.md").write_text("Root document content.")
        (subdir / "nested.md").write_text("Nested document content.")

        corpus = Corpus.from_folder(tmp_path)

        assert len(corpus) == 2
        # Check that document IDs preserve relative paths
        doc_ids = {doc.id for doc in corpus.documents}
        assert DocumentId("root.md") in doc_ids
        assert DocumentId("subdir/nested.md") in doc_ids

    def test_corpus_from_folder_with_custom_glob_pattern(self, tmp_path: Path) -> None:
        """Test Corpus.from_folder with custom glob pattern."""
        (tmp_path / "doc.md").write_text("Markdown content.")
        (tmp_path / "doc.txt").write_text("Text content.")

        corpus = Corpus.from_folder(tmp_path, glob_pattern="**/*.txt")

        assert len(corpus) == 1
        assert corpus.documents[0].id == DocumentId("doc.txt")

    def test_corpus_from_folder_nonexistent_folder(self) -> None:
        """Test Corpus.from_folder raises error for nonexistent folder."""
        with pytest.raises(FileNotFoundError, match="Folder not found"):
            Corpus.from_folder("/nonexistent/folder/path")

    def test_corpus_from_folder_empty_folder(self, tmp_path: Path) -> None:
        """Test Corpus.from_folder with folder containing no matching files."""
        corpus = Corpus.from_folder(tmp_path)

        assert len(corpus) == 0
        assert corpus.documents == []

    def test_corpus_from_folder_preserves_metadata(self, tmp_path: Path) -> None:
        """Test that from_folder sets appropriate metadata."""
        (tmp_path / "test.md").write_text("Test content")

        corpus = Corpus.from_folder(tmp_path, glob_pattern="*.md")

        assert "source_folder" in corpus.metadata
        assert "glob_pattern" in corpus.metadata
        assert corpus.metadata["glob_pattern"] == "*.md"
        # Each document should have source_path metadata
        for doc in corpus.documents:
            assert "source_path" in doc.metadata

    def test_corpus_get_document_existing(self) -> None:
        """Test get_document returns correct document for existing ID."""
        target_doc = Document(id=DocumentId("target.md"), content="Target content")
        docs = [
            Document(id=DocumentId("doc1.md"), content="Content 1"),
            target_doc,
            Document(id=DocumentId("doc3.md"), content="Content 3"),
        ]
        corpus = Corpus(documents=docs)

        result = corpus.get_document(DocumentId("target.md"))

        assert result is not None
        assert result.id == DocumentId("target.md")
        assert result.content == "Target content"

    def test_corpus_get_document_nonexistent(self) -> None:
        """Test get_document returns None for nonexistent ID."""
        docs = [
            Document(id=DocumentId("doc1.md"), content="Content 1"),
            Document(id=DocumentId("doc2.md"), content="Content 2"),
        ]
        corpus = Corpus(documents=docs)

        result = corpus.get_document(DocumentId("nonexistent.md"))

        assert result is None

    def test_corpus_get_document_empty_corpus(self) -> None:
        """Test get_document returns None for empty corpus."""
        corpus = Corpus(documents=[])

        result = corpus.get_document(DocumentId("any.md"))

        assert result is None

    def test_corpus_len(self) -> None:
        """Test __len__ returns correct document count."""
        docs = [Document(id=DocumentId(f"doc{i}.md"), content=f"Content {i}") for i in range(5)]
        corpus = Corpus(documents=docs)

        assert len(corpus) == 5

    def test_corpus_len_empty(self) -> None:
        """Test __len__ returns 0 for empty corpus."""
        corpus = Corpus(documents=[])

        assert len(corpus) == 0

    def test_corpus_immutability(self) -> None:
        """Test that Corpus is frozen (immutable)."""
        docs = [Document(id=DocumentId("doc.md"), content="Content")]
        corpus = Corpus(documents=docs)

        with pytest.raises(ValidationError):  # Pydantic frozen model raises ValidationError
            corpus.documents = []  # type: ignore[misc]
