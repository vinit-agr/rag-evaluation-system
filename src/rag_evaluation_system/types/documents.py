"""Document and Corpus models for the evaluation framework."""
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .primitives import DocumentId


class Document(BaseModel):
    """A source document from the corpus."""

    model_config = ConfigDict(frozen=True)

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
        """Load documents from a folder."""
        folder = Path(folder_path)
        documents: list[Document] = []

        for file_path in sorted(folder.glob(glob_pattern)):
            content = file_path.read_text(encoding="utf-8")
            doc_id = DocumentId(file_path.name)
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
