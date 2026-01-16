"""Token-level evaluation orchestrator for RAG pipelines.

This module provides the TokenLevelEvaluation class for evaluating retrieval
pipelines using token-level (character span) metrics (SpanRecall, SpanPrecision,
SpanIoU). Ground truth consists of character spans that should be retrieved for
each query.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, ClassVar

from rag_evaluation_system.chunkers.adapter import ChunkerPositionAdapter
from rag_evaluation_system.chunkers.base import Chunker, PositionAwareChunker
from rag_evaluation_system.evaluation.metrics import (
    SpanIoU,
    SpanPrecision,
    SpanRecall,
    TokenLevelMetric,
)
from rag_evaluation_system.types import (
    CharacterSpan,
    Corpus,
    DocumentId,
    EvaluationResult,
    PositionAwareChunk,
    Query,
    QueryId,
    QueryText,
    TokenLevelGroundTruth,
)
from rag_evaluation_system.vector_stores.chroma import ChromaVectorStore

if TYPE_CHECKING:
    from rag_evaluation_system.embedders.base import Embedder
    from rag_evaluation_system.rerankers.base import Reranker
    from rag_evaluation_system.vector_stores.base import VectorStore


class TokenLevelEvaluation:
    """Evaluation using token-level (character span) metrics.

    Compares character overlap between retrieved chunks and ground truth spans.
    Metrics are continuous: measures what fraction of relevant content was retrieved.

    This evaluation type is chunker-independent - the same ground truth dataset
    can be used to evaluate ANY chunking strategy, enabling fair comparison.

    IMPORTANT: The chunker must be a PositionAwareChunker (or will be wrapped
    with ChunkerPositionAdapter) because we need position information from
    chunks to compute overlap with ground truth character spans.

    Attributes:
        corpus: The document corpus to evaluate against.
        langsmith_dataset_name: Name of the LangSmith dataset containing ground truth.

    Example:
        >>> corpus = Corpus.from_folder("./docs")
        >>> eval = TokenLevelEvaluation(
        ...     corpus=corpus,
        ...     langsmith_dataset_name="my-token-dataset",
        ... )
        >>> result = eval.run(
        ...     chunker=RecursiveCharacterChunker(chunk_size=200),
        ...     embedder=OpenAIEmbedder(),
        ...     k=5,
        ... )
        >>> print(result.metrics)
        {'span_recall': 0.92, 'span_precision': 0.78, 'span_iou': 0.73}
    """

    DEFAULT_METRICS: ClassVar[list[TokenLevelMetric]] = [
        SpanRecall(),
        SpanPrecision(),
        SpanIoU(),
    ]

    def __init__(
        self,
        corpus: Corpus,
        langsmith_dataset_name: str,
    ) -> None:
        """Initialize the token-level evaluation.

        Args:
            corpus: The document corpus to evaluate against.
            langsmith_dataset_name: Name of the LangSmith dataset containing ground truth.
        """
        self.corpus = corpus
        self.langsmith_dataset_name = langsmith_dataset_name

    def run(
        self,
        chunker: Chunker | PositionAwareChunker,
        embedder: Embedder,
        k: int = 5,
        vector_store: VectorStore | None = None,
        reranker: Reranker | None = None,
        metrics: list[TokenLevelMetric] | None = None,
    ) -> EvaluationResult:
        """Run token-level evaluation.

        Pipeline:
        1. Wrap chunker with ChunkerPositionAdapter if needed
        2. Chunk corpus with positions
        3. Embed and index chunks in vector store
        4. For each query in dataset:
           - Retrieve top-k chunks
           - Optionally rerank results
           - Convert retrieved chunks to CharacterSpans
           - Compare retrieved spans vs ground truth spans (character overlap)
        5. Compute metrics (span recall, precision, IoU)

        Note on overlapping spans:
            Retrieved spans are merged before comparison. Each character
            is counted at most once to avoid inflating metrics.

        Args:
            chunker: Chunker to use. Will be wrapped with PositionAdapter if needed.
                MUST produce position-aware chunks for metric calculation.
            embedder: Embedder for generating vector representations.
            k: Number of chunks to retrieve per query. Defaults to 5.
            vector_store: Vector store for indexing/search. Defaults to ChromaVectorStore.
            reranker: Optional reranker to apply after retrieval.
            metrics: List of metrics to compute. Defaults to [SpanRecall, SpanPrecision, SpanIoU].

        Returns:
            EvaluationResult with computed metrics.
        """
        # Default vector store to ChromaDB if not provided
        if vector_store is None:
            vector_store = ChromaVectorStore()

        # Default metrics if not provided
        if metrics is None:
            metrics = self.DEFAULT_METRICS

        # Step 1: Wrap chunker if needed - MUST be position-aware for token-level eval
        position_aware_chunker = self._ensure_position_aware(chunker)

        # Step 2: Chunk corpus with positions
        all_chunks = self._chunk_corpus(position_aware_chunker)

        # Step 3: Embed chunk contents
        chunk_contents = [chunk.content for chunk in all_chunks]
        embeddings = embedder.embed(chunk_contents)

        # Step 4: Add to vector store
        vector_store.add(all_chunks, embeddings)

        # Step 5: Load ground truth from LangSmith
        ground_truths = self._load_ground_truth()

        # Step 6: Evaluate each query
        all_results: dict[str, list[float]] = {metric.name: [] for metric in metrics}

        for gt in ground_truths:
            # Embed query
            query_embedding = embedder.embed_query(gt.query.text)

            # Search vector store for k chunks
            retrieved_chunks = vector_store.search(query_embedding, k=k)

            # Optionally rerank
            if reranker is not None:
                retrieved_chunks = reranker.rerank(
                    gt.query.text,
                    retrieved_chunks,
                    top_k=k,
                )

            # Convert chunks to spans
            retrieved_spans = [chunk.to_span() for chunk in retrieved_chunks]

            # Calculate each metric
            for metric in metrics:
                score = metric.calculate(retrieved_spans, gt.relevant_spans)
                all_results[metric.name].append(score)

        # Step 7: Aggregate metrics (average)
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

    def _ensure_position_aware(
        self,
        chunker: Chunker | PositionAwareChunker,
    ) -> PositionAwareChunker:
        """Ensure the chunker is position-aware, wrapping if necessary.

        Args:
            chunker: The chunker to check/wrap.

        Returns:
            A PositionAwareChunker instance.
        """
        if isinstance(chunker, PositionAwareChunker):
            return chunker
        # Wrap with adapter if it's just a basic Chunker
        return ChunkerPositionAdapter(chunker)

    def _chunk_corpus(
        self,
        chunker: PositionAwareChunker,
    ) -> list[PositionAwareChunk]:
        """Chunk all documents in the corpus with position tracking.

        Args:
            chunker: The position-aware chunker to use.

        Returns:
            List of PositionAwareChunk objects from all documents.
        """
        all_chunks: list[PositionAwareChunk] = []

        for doc in self.corpus.documents:
            doc_chunks = chunker.chunk_with_positions(doc)
            all_chunks.extend(doc_chunks)

        return all_chunks

    def _load_ground_truth(self) -> list[TokenLevelGroundTruth]:
        """Load ground truth data from LangSmith.

        This method fetches the dataset from LangSmith and converts
        each example to a TokenLevelGroundTruth object.

        Returns:
            List of TokenLevelGroundTruth objects.

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

        ground_truths: list[TokenLevelGroundTruth] = []
        for example in examples:
            # Extract query from inputs
            inputs = example.inputs or {}
            query_text = inputs.get("query", "")

            # Extract relevant spans from outputs
            outputs = example.outputs or {}
            relevant_spans_raw = outputs.get("relevant_spans", [])

            # Convert to typed CharacterSpans
            relevant_spans: list[CharacterSpan] = []
            for span_dict in relevant_spans_raw:
                span = CharacterSpan(
                    doc_id=DocumentId(span_dict.get("doc_id", "")),
                    start=span_dict.get("start", 0),
                    end=span_dict.get("end", 0),
                    text=span_dict.get("text", ""),
                )
                relevant_spans.append(span)

            # Create Query object
            query_id = str(example.id) if example.id else str(uuid.uuid4())
            query = Query(
                id=QueryId(query_id),
                text=QueryText(query_text),
                metadata=example.metadata or {},
            )

            ground_truths.append(
                TokenLevelGroundTruth(
                    query=query,
                    relevant_spans=relevant_spans,
                )
            )

        return ground_truths


__all__ = ["TokenLevelEvaluation"]
