"""Chunk-level evaluation orchestrator for RAG pipelines.

This module provides the ChunkLevelEvaluation class for evaluating retrieval
pipelines using chunk-level metrics (Recall, Precision, F1). Ground truth
consists of chunk IDs that should be retrieved for each query.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, ClassVar

from rag_evaluation_system.evaluation.metrics import (
    ChunkF1,
    ChunkLevelMetric,
    ChunkPrecision,
    ChunkRecall,
)
from rag_evaluation_system.types import (
    ChunkId,
    ChunkLevelGroundTruth,
    Corpus,
    DocumentId,
    EvaluationResult,
    PositionAwareChunk,
    Query,
    QueryId,
    QueryText,
)
from rag_evaluation_system.utils.hashing import generate_chunk_id, generate_pa_chunk_id
from rag_evaluation_system.vector_stores.chroma import ChromaVectorStore

if TYPE_CHECKING:
    from rag_evaluation_system.chunkers.base import Chunker
    from rag_evaluation_system.embedders.base import Embedder
    from rag_evaluation_system.rerankers.base import Reranker
    from rag_evaluation_system.vector_stores.base import VectorStore


class ChunkLevelEvaluation:
    """Evaluation using chunk-level metrics.

    Compares retrieved chunk IDs against ground truth chunk IDs.
    Metrics are binary: a chunk is either relevant or not.

    This evaluation type is simpler but ties evaluation to a specific
    chunking strategy (the same chunker used for synthetic data generation
    must be used during evaluation).

    Attributes:
        corpus: The document corpus to evaluate against.
        langsmith_dataset_name: Name of the LangSmith dataset containing ground truth.

    Example:
        >>> corpus = Corpus.from_folder("./docs")
        >>> eval = ChunkLevelEvaluation(
        ...     corpus=corpus,
        ...     langsmith_dataset_name="my-chunk-dataset",
        ... )
        >>> result = eval.run(
        ...     chunker=RecursiveCharacterChunker(chunk_size=200),
        ...     embedder=OpenAIEmbedder(),
        ...     k=5,
        ... )
        >>> print(result.metrics)
        {'chunk_recall': 0.85, 'chunk_precision': 0.72, 'chunk_f1': 0.78}
    """

    DEFAULT_METRICS: ClassVar[list[ChunkLevelMetric]] = [
        ChunkRecall(),
        ChunkPrecision(),
        ChunkF1(),
    ]

    def __init__(
        self,
        corpus: Corpus,
        langsmith_dataset_name: str,
    ) -> None:
        """Initialize the chunk-level evaluation.

        Args:
            corpus: The document corpus to evaluate against.
            langsmith_dataset_name: Name of the LangSmith dataset containing ground truth.
        """
        self.corpus = corpus
        self.langsmith_dataset_name = langsmith_dataset_name

    def run(
        self,
        chunker: Chunker,
        embedder: Embedder,
        k: int = 5,
        vector_store: VectorStore | None = None,
        reranker: Reranker | None = None,
        metrics: list[ChunkLevelMetric] | None = None,
    ) -> EvaluationResult:
        """Run chunk-level evaluation.

        Pipeline:
        1. Chunk corpus using chunker
        2. Generate chunk IDs (content hash with "chunk_" prefix)
        3. Embed and index chunks in vector store
        4. For each query in dataset:
           - Retrieve top-k chunks
           - Optionally rerank results
           - Compare retrieved chunk IDs vs ground truth chunk IDs
        5. Compute metrics (recall, precision, F1)

        Args:
            chunker: Chunker to use for splitting documents.
            embedder: Embedder for generating vector representations.
            k: Number of chunks to retrieve per query. Defaults to 5.
            vector_store: Vector store for indexing/search. Defaults to ChromaVectorStore.
            reranker: Optional reranker to apply after retrieval.
            metrics: List of metrics to compute. Defaults to [ChunkRecall, ChunkPrecision, ChunkF1].

        Returns:
            EvaluationResult with computed metrics.
        """
        # Default vector store to ChromaDB if not provided
        if vector_store is None:
            vector_store = ChromaVectorStore()

        # Default metrics if not provided
        if metrics is None:
            metrics = self.DEFAULT_METRICS

        # Step 1-2: Chunk corpus and generate IDs
        chunks, chunk_ids = self._chunk_corpus(chunker)

        # Step 3: Embed chunks
        embeddings = embedder.embed(chunks)

        # Step 4: Convert to PositionAwareChunks for vector store compatibility
        pa_chunks = self._to_position_aware(chunks, chunk_ids)

        # Step 5: Add to vector store
        vector_store.add(pa_chunks, embeddings)

        # Step 6: Load ground truth from LangSmith
        ground_truths = self._load_ground_truth()

        # Step 7: Evaluate each query
        all_results: dict[str, list[float]] = {metric.name: [] for metric in metrics}

        for gt in ground_truths:
            # Embed query
            query_embedding = embedder.embed_query(gt.query.text)

            # Search vector store for k chunks
            retrieved_pa_chunks = vector_store.search(query_embedding, k=k)

            # Optionally rerank
            if reranker is not None:
                retrieved_pa_chunks = reranker.rerank(
                    gt.query.text,
                    retrieved_pa_chunks,
                    top_k=k,
                )

            # Convert IDs back to ChunkIds (replace "pa_chunk_" with "chunk_")
            retrieved_chunk_ids = self._convert_to_chunk_ids(retrieved_pa_chunks)

            # Calculate each metric
            for metric in metrics:
                score = metric.calculate(retrieved_chunk_ids, gt.relevant_chunk_ids)
                all_results[metric.name].append(score)

        # Step 8: Aggregate metrics (average)
        aggregated_metrics: dict[str, float] = {}
        for metric_name, scores in all_results.items():
            if scores:
                aggregated_metrics[metric_name] = sum(scores) / len(scores)
            else:
                aggregated_metrics[metric_name] = 0.0

        return EvaluationResult(
            metrics=aggregated_metrics,
            experiment_url=None,
            raw_results=all_results,
        )

    def _chunk_corpus(self, chunker: Chunker) -> tuple[list[str], list[ChunkId]]:
        """Chunk all documents in the corpus.

        Args:
            chunker: The chunker to use.

        Returns:
            A tuple of (chunk_texts, chunk_ids).
        """
        all_chunks: list[str] = []
        all_chunk_ids: list[ChunkId] = []

        for doc in self.corpus.documents:
            doc_chunks = chunker.chunk(doc.content)
            for chunk_text in doc_chunks:
                all_chunks.append(chunk_text)
                all_chunk_ids.append(generate_chunk_id(chunk_text))

        return all_chunks, all_chunk_ids

    def _to_position_aware(
        self,
        chunks: list[str],
        chunk_ids: list[ChunkId],
    ) -> list[PositionAwareChunk]:
        """Convert chunks to PositionAwareChunks with placeholder positions.

        The vector store interface requires PositionAwareChunk objects.
        For chunk-level evaluation, we don't need actual positions, so we
        use placeholder values. The important part is preserving the
        relationship between chunk content and ID.

        Args:
            chunks: List of chunk text strings.
            chunk_ids: List of corresponding ChunkIds.

        Returns:
            List of PositionAwareChunk objects with placeholder positions.
        """
        pa_chunks: list[PositionAwareChunk] = []

        for chunk_text, chunk_id in zip(chunks, chunk_ids, strict=True):
            # Convert chunk_xxx to pa_chunk_xxx for vector store
            pa_chunk_id = generate_pa_chunk_id(chunk_text)

            # Use placeholder positions and doc_id since chunk-level eval
            # only cares about chunk identity, not position
            pa_chunks.append(
                PositionAwareChunk(
                    id=pa_chunk_id,
                    content=chunk_text,
                    doc_id=DocumentId("__placeholder__"),
                    start=0,
                    end=len(chunk_text),
                    metadata={"original_chunk_id": chunk_id},
                )
            )

        return pa_chunks

    def _convert_to_chunk_ids(
        self,
        pa_chunks: list[PositionAwareChunk],
    ) -> list[ChunkId]:
        """Convert PositionAwareChunk IDs back to ChunkIds.

        Since both ID types are based on content hash, we replace the
        "pa_chunk_" prefix with "chunk_" to get the equivalent ChunkId.

        Args:
            pa_chunks: List of PositionAwareChunk objects.

        Returns:
            List of corresponding ChunkIds.
        """
        chunk_ids: list[ChunkId] = []
        for pa_chunk in pa_chunks:
            # Replace "pa_chunk_" prefix with "chunk_"
            chunk_id_str = pa_chunk.id.replace("pa_chunk_", "chunk_")
            chunk_ids.append(ChunkId(chunk_id_str))
        return chunk_ids

    def _load_ground_truth(self) -> list[ChunkLevelGroundTruth]:
        """Load ground truth data from LangSmith.

        This method fetches the dataset from LangSmith and converts
        each example to a ChunkLevelGroundTruth object.

        Returns:
            List of ChunkLevelGroundTruth objects.

        Raises:
            ImportError: If langsmith package is not installed.
        """
        try:
            from langsmith import Client
        except ImportError as e:
            raise ImportError(
                "langsmith package is required for loading ground truth. "
                "Install it with: pip install langsmith"
            ) from e

        client = Client()
        dataset = client.read_dataset(dataset_name=self.langsmith_dataset_name)
        examples = list(client.list_examples(dataset_id=dataset.id))

        ground_truths: list[ChunkLevelGroundTruth] = []
        for example in examples:
            # Extract query from inputs
            inputs = example.inputs or {}
            query_text = inputs.get("query", "")

            # Extract relevant chunk IDs from outputs
            outputs = example.outputs or {}
            relevant_chunk_ids_raw = outputs.get("relevant_chunk_ids", [])

            # Convert to typed ChunkIds
            relevant_chunk_ids = [ChunkId(cid) for cid in relevant_chunk_ids_raw]

            # Create Query object
            query_id = str(example.id) if example.id else str(uuid.uuid4())
            query = Query(
                id=QueryId(query_id),
                text=QueryText(query_text),
                metadata=example.metadata or {},
            )

            ground_truths.append(
                ChunkLevelGroundTruth(
                    query=query,
                    relevant_chunk_ids=relevant_chunk_ids,
                )
            )

        return ground_truths


__all__ = ["ChunkLevelEvaluation"]
