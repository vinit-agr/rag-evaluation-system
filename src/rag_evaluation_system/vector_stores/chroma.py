"""ChromaDB vector store implementation.

This module provides a VectorStore implementation using ChromaDB for
persistent vector storage and similarity search.
Requires the `chroma` optional dependency: `pip install rag-evaluation-system[chroma]`
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, cast

from rag_evaluation_system.types.chunks import PositionAwareChunk
from rag_evaluation_system.types.primitives import DocumentId, PositionAwareChunkId
from rag_evaluation_system.vector_stores.base import VectorStore

if TYPE_CHECKING:
    import chromadb
    from chromadb.api.models.Collection import Collection
    from chromadb.api.types import Embedding, Metadata


class ChromaVectorStore(VectorStore):
    """VectorStore implementation using ChromaDB.

    ChromaDB is an open-source embedding database that provides efficient
    vector similarity search with optional persistence.

    Attributes:
        collection_name: Name of the ChromaDB collection.
        persist_directory: Optional directory for persistent storage.
        _client: The ChromaDB client instance.
        _collection: The ChromaDB collection for storing embeddings.

    Example:
        >>> store = ChromaVectorStore()  # In-memory storage
        >>> store.add(chunks, embeddings)
        >>> results = store.search(query_embedding, k=5)

        >>> store = ChromaVectorStore(persist_directory="./chroma_data")  # Persistent
        >>> store.add(chunks, embeddings)
        >>> # Data persists across restarts
    """

    def __init__(
        self,
        collection_name: str | None = None,
        persist_directory: str | None = None,
    ) -> None:
        """Initialize the ChromaDB vector store.

        Args:
            collection_name: Name for the ChromaDB collection. If not provided,
                a unique name will be generated using UUID.
            persist_directory: Optional directory path for persistent storage.
                If not provided, data is stored in-memory only.

        Raises:
            ImportError: If the chromadb package is not installed.
        """
        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "chromadb package is required for ChromaVectorStore. "
                "Install it with: pip install rag-evaluation-system[chroma] "
                "or: pip install chromadb"
            ) from e

        self.collection_name = collection_name or f"rag_eval_{uuid.uuid4().hex[:8]}"
        self.persist_directory = persist_directory

        # Create client based on persistence setting
        if persist_directory is not None:
            self._client: chromadb.ClientAPI = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        # Get or create collection with cosine similarity
        self._collection: Collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def name(self) -> str:
        """Return a descriptive name for this vector store.

        Returns:
            A string describing this ChromaDB configuration.
        """
        if self.persist_directory:
            return f"Chroma({self.collection_name}, persistent={self.persist_directory})"
        return f"Chroma({self.collection_name}, in-memory)"

    def add(
        self,
        chunks: list[PositionAwareChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add chunks and their embeddings to the store.

        Args:
            chunks: List of position-aware chunks to store.
            embeddings: List of embedding vectors corresponding to each chunk.

        Raises:
            ValueError: If chunks and embeddings have different lengths.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )

        if not chunks:
            return

        # Prepare data for ChromaDB
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[Metadata] = []
        chroma_embeddings: list[Embedding] = []

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            ids.append(chunk.id)
            documents.append(chunk.content)
            # Cast to Embedding type (Sequence[float])
            chroma_embeddings.append(cast("Embedding", embedding))
            metadatas.append(
                {
                    "doc_id": chunk.doc_id,
                    "start": chunk.start,
                    "end": chunk.end,
                    # Store original metadata as JSON-compatible values
                    **{
                        k: v
                        for k, v in chunk.metadata.items()
                        if isinstance(v, (str, int, float, bool))
                    },
                }
            )

        self._collection.add(
            ids=ids,
            embeddings=chroma_embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[PositionAwareChunk]:
        """Search for the k most similar chunks.

        Args:
            query_embedding: The query embedding vector.
            k: Number of results to return. Defaults to 5.

        Returns:
            List of the k most similar PositionAwareChunk objects,
            ordered by decreasing similarity.
        """
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas"],
        )

        # Reconstruct PositionAwareChunk objects from results
        chunks: list[PositionAwareChunk] = []

        # Handle potential None values from ChromaDB
        result_ids = results.get("ids")
        result_documents = results.get("documents")
        result_metadatas = results.get("metadatas")

        ids_list: list[str] = result_ids[0] if result_ids else []
        documents_list: list[str | None] = result_documents[0] if result_documents else []
        metadatas_list: list[Metadata | None] = result_metadatas[0] if result_metadatas else []

        for chunk_id, content, metadata in zip(
            ids_list, documents_list, metadatas_list, strict=False
        ):
            if content is None or metadata is None:
                continue

            # Extract position metadata
            doc_id = metadata.get("doc_id", "")
            start = metadata.get("start", 0)
            end = metadata.get("end", 0)

            # Reconstruct additional metadata (excluding position fields)
            extra_metadata: dict[str, Any] = {
                k: v for k, v in metadata.items() if k not in ("doc_id", "start", "end")
            }

            chunk = PositionAwareChunk(
                id=PositionAwareChunkId(chunk_id),
                content=content,
                doc_id=DocumentId(str(doc_id)),
                start=int(start) if start is not None else 0,
                end=int(end) if end is not None else 0,
                metadata=extra_metadata,
            )
            chunks.append(chunk)

        return chunks

    def clear(self) -> None:
        """Remove all chunks and embeddings from the store.

        Deletes and recreates the collection to ensure a clean state.
        """
        self._client.delete_collection(name=self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


__all__ = ["ChromaVectorStore"]
