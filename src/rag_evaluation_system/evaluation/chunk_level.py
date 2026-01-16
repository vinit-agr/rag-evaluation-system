"""Chunk-level evaluation orchestrator."""
import logging

from rag_evaluation_system.chunkers.base import Chunker
from rag_evaluation_system.embedders.base import Embedder
from rag_evaluation_system.rerankers.base import Reranker
from rag_evaluation_system.types import (
    ChunkId,
    ChunkLevelGroundTruth,
    Corpus,
    EvaluationResult,
)
from rag_evaluation_system.utils.hashing import generate_chunk_id
from rag_evaluation_system.vector_stores.base import VectorStore
from .metrics.base import ChunkLevelMetric
from .metrics.chunk_level import ChunkF1, ChunkPrecision, ChunkRecall

logger = logging.getLogger(__name__)


class ChunkLevelEvaluation:
    """Evaluation using chunk-level metrics."""

    DEFAULT_METRICS: list[ChunkLevelMetric] = [
        ChunkRecall(),
        ChunkPrecision(),
        ChunkF1(),
    ]

    def __init__(self, corpus: Corpus, langsmith_dataset_name: str):
        self._corpus = corpus
        self._dataset_name = langsmith_dataset_name

    def run(
        self,
        chunker: Chunker,
        embedder: Embedder,
        k: int = 5,
        vector_store: VectorStore | None = None,
        reranker: Reranker | None = None,
        metrics: list[ChunkLevelMetric] | None = None,
    ) -> EvaluationResult:
        if vector_store is None:
            from rag_evaluation_system.vector_stores.chroma import ChromaVectorStore

            vector_store = ChromaVectorStore()

        metrics = metrics or self.DEFAULT_METRICS

        logger.info("Chunking corpus with %s", chunker.name)
        chunks, chunk_ids = self._chunk_corpus(chunker)

        logger.info("Embedding %s chunks with %s", len(chunks), embedder.name)
        embeddings = embedder.embed(chunks)

        logger.info("Indexing in %s", vector_store.name)
        pa_chunks = self._to_position_aware(chunks, chunk_ids)
        vector_store.add(pa_chunks, embeddings)

        ground_truth = self._load_ground_truth()

        logger.info("Evaluating %s queries", len(ground_truth))
        all_results: dict[str, list[float]] = {m.name: [] for m in metrics}

        for gt in ground_truth:
            query_embedding = embedder.embed_query(gt.query.text)
            retrieved_chunks = vector_store.search(query_embedding, k)

            if reranker:
                retrieved_chunks = reranker.rerank(gt.query.text, retrieved_chunks, top_k=k)

            retrieved_ids = [
                ChunkId(str(c.id).replace("pa_chunk_", "chunk_")) for c in retrieved_chunks
            ]

            for metric in metrics:
                score = metric.calculate(retrieved_ids, gt.relevant_chunk_ids)
                all_results[metric.name].append(score)

        avg_metrics = {
            name: sum(scores) / len(scores) if scores else 0.0
            for name, scores in all_results.items()
        }

        logger.info("Results: %s", avg_metrics)

        return EvaluationResult(
            metrics=avg_metrics,
            experiment_url=None,
        )

    def _chunk_corpus(self, chunker: Chunker) -> tuple[list[str], list[ChunkId]]:
        chunks: list[str] = []
        chunk_ids: list[ChunkId] = []

        for doc in self._corpus.documents:
            doc_chunks = chunker.chunk(doc.content)
            for chunk_text in doc_chunks:
                chunks.append(chunk_text)
                chunk_ids.append(generate_chunk_id(chunk_text))

        return chunks, chunk_ids

    def _to_position_aware(self, chunks: list[str], chunk_ids: list[ChunkId]) -> list:
        from rag_evaluation_system.types import DocumentId, PositionAwareChunk, PositionAwareChunkId

        return [
            PositionAwareChunk(
                id=PositionAwareChunkId(str(cid).replace("chunk_", "pa_chunk_")),
                content=text,
                doc_id=DocumentId("unknown"),
                start=0,
                end=len(text),
            )
            for text, cid in zip(chunks, chunk_ids)
        ]

    def _load_ground_truth(self) -> list[ChunkLevelGroundTruth]:
        from rag_evaluation_system.langsmith.client import load_chunk_level_dataset

        return load_chunk_level_dataset(self._dataset_name)
