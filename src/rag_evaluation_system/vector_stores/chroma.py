"""ChromaDB vector store implementation."""
from typing import TYPE_CHECKING
import uuid

from rag_evaluation_system.types import DocumentId, PositionAwareChunk, PositionAwareChunkId
from .base import VectorStore

if TYPE_CHECKING:
    import chromadb


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store."""

    def __init__(
        self,
        collection_name: str | None = None,
        persist_directory: str | None = None,
    ):
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

    def add(self, chunks: list[PositionAwareChunk], embeddings: list[list[float]]) -> None:
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

    def search(self, query_embedding: list[float], k: int = 5) -> list[PositionAwareChunk]:
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
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
