"""Chunk-level synthetic data generator."""
import json
import logging
from typing import Any

from pydantic import BaseModel

from rag_evaluation_system.chunkers.base import Chunker
from rag_evaluation_system.types import (
    ChunkId,
    ChunkLevelGroundTruth,
    Corpus,
    Query,
    QueryId,
    QueryText,
)
from rag_evaluation_system.utils.hashing import generate_chunk_id
from ..base import SyntheticDatasetGenerator

logger = logging.getLogger(__name__)


class GeneratedQAPair(BaseModel):
    """A generated query with relevant chunk IDs."""

    query: str
    relevant_chunk_ids: list[str]


class ChunkLevelSyntheticDatasetGenerator(SyntheticDatasetGenerator):
    """Generate synthetic QA pairs with chunk-level ground truth."""

    SYSTEM_PROMPT = """You are an expert at generating evaluation data for RAG systems.
Given chunks from a document with their IDs, generate questions that can be
answered using specific chunks. For each question, list the chunk IDs that
contain the answer.

Output JSON format:
{
    "qa_pairs": [
        {
            "query": "What is...?",
            "relevant_chunk_ids": ["chunk_xxx", "chunk_yyy"]
        }
    ]
}"""

    def __init__(self, llm_client: Any, corpus: Corpus, chunker: Chunker):
        super().__init__(llm_client, corpus)
        self._chunker = chunker
        self._chunk_index: dict[ChunkId, str] = {}

    def generate(
        self,
        queries_per_doc: int = 5,
        upload_to_langsmith: bool = True,
        dataset_name: str | None = None,
    ) -> list[ChunkLevelGroundTruth]:
        logger.info("Chunking documents...")
        all_chunks = self._build_chunk_index()

        ground_truth: list[ChunkLevelGroundTruth] = []

        for doc in self._corpus.documents:
            doc_chunks = list(all_chunks.items())

            logger.info("Generating %s queries for %s", queries_per_doc, doc.id)
            qa_pairs = self._generate_qa_pairs(doc_chunks, queries_per_doc)

            for qa in qa_pairs:
                valid_ids = [
                    ChunkId(cid) for cid in qa.relevant_chunk_ids if ChunkId(cid) in self._chunk_index
                ]

                if not valid_ids:
                    logger.warning("No valid chunk IDs for query: %s...", qa.query[:50])
                    continue

                ground_truth.append(
                    ChunkLevelGroundTruth(
                        query=Query(
                            id=QueryId(f"q_{len(ground_truth)}"),
                            text=QueryText(qa.query),
                            metadata={"source_doc": str(doc.id)},
                        ),
                        relevant_chunk_ids=valid_ids,
                    )
                )

        if upload_to_langsmith:
            self._upload_to_langsmith(ground_truth, dataset_name)

        return ground_truth

    def _build_chunk_index(self) -> dict[ChunkId, str]:
        for doc in self._corpus.documents:
            chunks = self._chunker.chunk(doc.content)
            for chunk_text in chunks:
                chunk_id = generate_chunk_id(chunk_text)
                self._chunk_index[chunk_id] = chunk_text
        return self._chunk_index

    def _generate_qa_pairs(
        self,
        chunks: list[tuple[ChunkId, str]],
        num_queries: int,
    ) -> list[GeneratedQAPair]:
        chunk_text = "\n".join(
            f"[{chunk_id}]: {content[:500]}..." for chunk_id, content in chunks[:20]
        )

        prompt = f"""Here are chunks from a document:

{chunk_text}

Generate {num_queries} diverse questions that can be answered using these chunks.
For each question, list the chunk IDs that contain the answer."""

        response = self._call_llm(prompt)
        return self._parse_response(response)

    def _call_llm(self, prompt: str) -> str:
        response = self._llm.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _parse_response(self, response: str) -> list[GeneratedQAPair]:
        data = json.loads(response)
        return [GeneratedQAPair(**qa) for qa in data.get("qa_pairs", [])]

    def _upload_to_langsmith(
        self,
        ground_truth: list[ChunkLevelGroundTruth],
        dataset_name: str | None,
    ) -> None:
        from rag_evaluation_system.langsmith.upload import upload_chunk_level_dataset

        upload_chunk_level_dataset(ground_truth, dataset_name)
