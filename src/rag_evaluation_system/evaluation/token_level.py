"""Token-level evaluation orchestrator."""
import logging
from typing import Union

from rag_evaluation_system.chunkers.adapter import ChunkerPositionAdapter
from rag_evaluation_system.chunkers.base import Chunker, PositionAwareChunker
from rag_evaluation_system.embedders.base import Embedder
from rag_evaluation_system.rerankers.base import Reranker
from rag_evaluation_system.types import Corpus, EvaluationResult, TokenLevelGroundTruth
from rag_evaluation_system.vector_stores.base import VectorStore
from .metrics.base import TokenLevelMetric
from .metrics.token_level import SpanIoU, SpanPrecision, SpanRecall

logger = logging.getLogger(__name__)


class TokenLevelEvaluation:
    """Evaluation using token-level (character span) metrics."""

    DEFAULT_METRICS: list[TokenLevelMetric] = [
        SpanRecall(),
        SpanPrecision(),
        SpanIoU(),
    ]

    def __init__(self, corpus: Corpus, langsmith_dataset_name: str):
        self._corpus = corpus
        self._dataset_name = langsmith_dataset_name

    def run(
        self,
        chunker: Union[Chunker, PositionAwareChunker],
        embedder: Embedder,
        k: int = 5,
        vector_store: VectorStore | None = None,
        reranker: Reranker | None = None,
        metrics: list[TokenLevelMetric] | None = None,
    ) -> EvaluationResult:
        if vector_store is None:
            from rag_evaluation_system.vector_stores.chroma import ChromaVectorStore

            vector_store = ChromaVectorStore()

        metrics = metrics or self.DEFAULT_METRICS

        if isinstance(chunker, Chunker) and not isinstance(chunker, PositionAwareChunker):
            logger.info("Wrapping %s with position adapter", chunker.name)
            pa_chunker: PositionAwareChunker = ChunkerPositionAdapter(chunker)
        else:
            pa_chunker = chunker  # type: ignore[assignment]

        logger.info("Chunking corpus with %s", pa_chunker.name)
        all_chunks = []
        for doc in self._corpus.documents:
            doc_chunks = pa_chunker.chunk_with_positions(doc)
            all_chunks.extend(doc_chunks)

        logger.info("Generated %s position-aware chunks", len(all_chunks))

        logger.info("Embedding chunks with %s", embedder.name)
        chunk_texts = [c.content for c in all_chunks]
        embeddings = embedder.embed(chunk_texts)

        logger.info("Indexing in %s", vector_store.name)
        vector_store.add(all_chunks, embeddings)

        ground_truth = self._load_ground_truth()

        logger.info("Evaluating %s queries", len(ground_truth))
        all_results: dict[str, list[float]] = {m.name: [] for m in metrics}

        for gt in ground_truth:
            query_embedding = embedder.embed_query(gt.query.text)
            retrieved_chunks = vector_store.search(query_embedding, k)

            if reranker:
                retrieved_chunks = reranker.rerank(gt.query.text, retrieved_chunks, top_k=k)

            retrieved_spans = [chunk.to_span() for chunk in retrieved_chunks]

            for metric in metrics:
                score = metric.calculate(retrieved_spans, gt.relevant_spans)
                all_results[metric.name].append(score)

        avg_metrics = {
            name: sum(scores) / len(scores) if scores else 0.0
            for name, scores in all_results.items()
        }

        logger.info("Results: %s", avg_metrics)

        return EvaluationResult(metrics=avg_metrics)

    def _load_ground_truth(self) -> list[TokenLevelGroundTruth]:
        from rag_evaluation_system.langsmith.client import load_token_level_dataset

        return load_token_level_dataset(self._dataset_name)
