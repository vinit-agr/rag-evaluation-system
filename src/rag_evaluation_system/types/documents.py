"""Document and corpus models for the RAG evaluation system.

This module defines the foundational data structures for representing
source documents and document collections.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from rag_evaluation_system.types.primitives import DocumentId


class Document(BaseModel):
    """A source document from the corpus.

    Represents a single text file (typically markdown) that will be chunked
    and indexed for retrieval evaluation.

    Attributes:
        id: Unique identifier for this document. Used to reference the document
            in chunk IDs and ground truth data. Typically derived from filename.
        content: The full text content of the document.
        metadata: Arbitrary key-value pairs for additional document information.
            Examples: {"author": "John", "date": "2024-01-15", "source": "wiki"}
    """

    model_config = ConfigDict(frozen=True)

    id: DocumentId
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def char_count(self) -> int:
        """Return the number of characters in the document content."""
        return len(self.content)


class Corpus(BaseModel):
    """Collection of documents to evaluate against.

    The corpus represents the entire knowledge base that will be chunked,
    embedded, and indexed. Synthetic queries are generated from this corpus,
    and retrieval performance is measured against it.

    Attributes:
        documents: List of all documents in the corpus.
        metadata: Arbitrary key-value pairs for corpus-level information.
            Examples: {"name": "product_docs", "version": "2.0"}
    """

    model_config = ConfigDict(frozen=True)

    documents: list[Document]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_folder(
        cls,
        folder_path: str | Path,
        glob_pattern: str = "**/*.md",
    ) -> "Corpus":
        """Load all markdown files from a folder into a Corpus.

        Args:
            folder_path: Path to the folder containing documents.
            glob_pattern: Glob pattern for matching files. Default matches all
                markdown files recursively.

        Returns:
            A Corpus containing all matched documents.

        Raises:
            FileNotFoundError: If the folder does not exist.
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        documents: list[Document] = []
        for file_path in sorted(folder.glob(glob_pattern)):
            if file_path.is_file():
                content = file_path.read_text(encoding="utf-8")
                # Use relative path from folder as document ID
                relative_path = file_path.relative_to(folder)
                doc_id = DocumentId(str(relative_path))
                documents.append(
                    Document(
                        id=doc_id,
                        content=content,
                        metadata={"source_path": str(file_path)},
                    )
                )

        return cls(
            documents=documents,
            metadata={"source_folder": str(folder), "glob_pattern": glob_pattern},
        )

    def get_document(self, doc_id: DocumentId) -> Document | None:
        """Retrieve a document by its ID.

        Args:
            doc_id: The document ID to search for.

        Returns:
            The matching Document, or None if not found.
        """
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None

    def __len__(self) -> int:
        """Return the number of documents in the corpus."""
        return len(self.documents)


__all__ = [
    "Corpus",
    "Document",
]
