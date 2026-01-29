"""Chunk-level synthetic data generator.

This module implements synthetic QA pair generation with chunk-level ground truth.
The generator requires a chunker because chunk IDs must exist before we can
reference them in ground truth.
"""

import hashlib
import json
import logging
from typing import Any

from pydantic import BaseModel, ConfigDict

from rag_evaluation_system.chunkers.base import Chunker
from rag_evaluation_system.synthetic_datagen.base import SyntheticDatasetGenerator
from rag_evaluation_system.types.chunks import Chunk
from rag_evaluation_system.types.documents import Corpus, Document
from rag_evaluation_system.types.ground_truth import ChunkLevelGroundTruth
from rag_evaluation_system.types.primitives import ChunkId, QueryId, QueryText
from rag_evaluation_system.types.queries import Query
from rag_evaluation_system.utils.hashing import generate_chunk_id

logger = logging.getLogger(__name__)


class GeneratedQAPair(BaseModel):
    """A generated question-answer pair with chunk ID citations.

    Attributes:
        query: The generated question text.
        relevant_chunk_ids: List of chunk IDs that contain the answer.
    """

    model_config = ConfigDict(frozen=True)

    query: str
    relevant_chunk_ids: list[str]


class ChunkLevelSyntheticDatasetGenerator(SyntheticDatasetGenerator):
    """Generate synthetic QA pairs with chunk-level ground truth.

    This generator requires a chunker because chunk IDs must exist before
    we can reference them in ground truth. The LLM generates queries AND
    identifies relevant chunks simultaneously (chunk-level citation).

    Example:
        >>> from openai import OpenAI
        >>> generator = ChunkLevelSyntheticDatasetGenerator(
        ...     llm_client=OpenAI(),
        ...     corpus=corpus,
        ...     chunker=RecursiveCharacterChunker(chunk_size=200),
        ... )
        >>> ground_truth = generator.generate(queries_per_doc=5)
    """

    SYSTEM_PROMPT = """You are a helpful assistant that generates question-answer pairs for evaluating retrieval systems.

Given a set of text chunks with their IDs, generate questions that can be answered using the information in specific chunks.

Rules:
1. Generate diverse questions that cover different aspects of the content
2. Questions should be answerable using ONLY the provided chunks
3. Each question should cite the specific chunk IDs that contain the answer
4. Questions should be clear, specific, and unambiguous
5. Avoid trivial questions or questions that require external knowledge

Output Format:
Return a JSON array of objects, each with:
- "query": The question text
- "relevant_chunk_ids": Array of chunk IDs that contain the answer

Example output:
[
  {"query": "What is the main benefit of RAG systems?", "relevant_chunk_ids": ["chunk_abc123def456"]},
  {"query": "How does retrieval augmentation reduce hallucination?", "relevant_chunk_ids": ["chunk_abc123def456", "chunk_789xyz012345"]}
]

Important: Only use chunk IDs that were provided to you. Do not invent chunk IDs."""

    def __init__(
        self,
        llm_client: Any,
        corpus: Corpus,
        chunker: Chunker,
    ) -> None:
        """Initialize the chunk-level generator.

        Args:
            llm_client: An LLM client with an OpenAI-compatible interface.
            corpus: The document corpus to generate synthetic data from.
            chunker: The chunker to use for splitting documents into chunks.
                Ground truth will be tied to this specific chunker.
        """
        super().__init__(llm_client, corpus)
        self._chunker = chunker
        self._chunk_index: dict[ChunkId, str] = {}

    def generate(
        self,
        queries_per_doc: int = 5,
        upload_to_langsmith: bool = True,
        dataset_name: str | None = None,
    ) -> list[ChunkLevelGroundTruth]:
        """Generate synthetic queries with relevant chunk IDs.

        Process:
        1. Chunk all documents, build chunk index with IDs
        2. For each document's chunks:
           a. Present chunks with their IDs to the LLM
           b. Ask LLM to generate queries that can be answered by specific chunks
           c. LLM returns both the query AND the relevant chunk IDs (citations)
        3. Validate that returned chunk IDs exist
        4. Upload to LangSmith if requested

        Args:
            queries_per_doc: Number of queries to generate per document.
            upload_to_langsmith: Whether to upload the dataset to LangSmith.
            dataset_name: Name for the LangSmith dataset. If None, auto-generated.

        Returns:
            List of ChunkLevelGroundTruth objects containing queries and
            their relevant chunk IDs.
        """
        # Build chunk index from all documents
        self._build_chunk_index()

        ground_truth_list: list[ChunkLevelGroundTruth] = []

        for doc in self._corpus.documents:
            # Get chunks for this document
            doc_chunks = self._get_document_chunks(doc)
            if not doc_chunks:
                logger.warning(f"No chunks generated for document: {doc.id}")
                continue

            # Generate QA pairs for this document's chunks
            qa_pairs = self._generate_qa_pairs(doc_chunks, queries_per_doc)

            # Convert to ground truth objects
            for qa_pair in qa_pairs:
                # Validate and filter chunk IDs
                valid_chunk_ids = self._validate_chunk_ids(qa_pair.relevant_chunk_ids)
                if not valid_chunk_ids:
                    logger.warning(f"No valid chunk IDs for query: {qa_pair.query[:50]}...")
                    continue

                # Generate query ID from content hash
                query_id = QueryId(
                    f"query_{hashlib.sha256(qa_pair.query.encode()).hexdigest()[:12]}"
                )

                ground_truth = ChunkLevelGroundTruth(
                    query=Query(
                        id=query_id,
                        text=QueryText(qa_pair.query),
                        metadata={"source_doc": doc.id},
                    ),
                    relevant_chunk_ids=valid_chunk_ids,
                )
                ground_truth_list.append(ground_truth)

        # Upload to LangSmith if requested
        if upload_to_langsmith and ground_truth_list:
            self._upload_to_langsmith(ground_truth_list, dataset_name)

        return ground_truth_list

    def _build_chunk_index(self) -> dict[ChunkId, str]:
        """Chunk all documents and build an index of chunk ID to content.

        Returns:
            Dictionary mapping chunk IDs to their text content.
        """
        self._chunk_index.clear()

        for doc in self._corpus.documents:
            chunk_texts = self._chunker.chunk(doc.content)
            for chunk_text in chunk_texts:
                chunk_id = generate_chunk_id(chunk_text)
                self._chunk_index[chunk_id] = chunk_text

        logger.info(f"Built chunk index with {len(self._chunk_index)} chunks")
        return self._chunk_index

    def _get_document_chunks(self, doc: Document) -> list[Chunk]:
        """Get all chunks for a specific document.

        Args:
            doc: The document to chunk.

        Returns:
            List of Chunk objects for this document.
        """
        chunk_texts = self._chunker.chunk(doc.content)
        chunks: list[Chunk] = []

        for chunk_text in chunk_texts:
            chunk_id = generate_chunk_id(chunk_text)
            chunks.append(
                Chunk(
                    id=chunk_id,
                    content=chunk_text,
                    doc_id=doc.id,
                )
            )

        return chunks

    def _generate_qa_pairs(self, chunks: list[Chunk], num_queries: int) -> list[GeneratedQAPair]:
        """Generate QA pairs for a set of chunks.

        Args:
            chunks: List of chunks to generate questions about.
            num_queries: Number of questions to generate.

        Returns:
            List of GeneratedQAPair objects.
        """
        # Format chunks for the prompt
        chunks_text = "\n\n".join(f"[{chunk.id}]: {chunk.content}" for chunk in chunks)

        user_prompt = f"""Here are the text chunks with their IDs:

{chunks_text}

Generate exactly {num_queries} diverse questions that can be answered using these chunks.
For each question, list the chunk IDs that contain the relevant information.

Return ONLY a valid JSON array."""

        # Call LLM
        response = self._call_llm(user_prompt)

        # Parse response
        return self._parse_response(response)

    def _call_llm(self, user_prompt: str) -> str:
        """Call the LLM with the system prompt and user prompt.

        Args:
            user_prompt: The user message to send.

        Returns:
            The LLM's response text.
        """
        response = self._llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content or ""

    def _parse_response(self, response: str) -> list[GeneratedQAPair]:
        """Parse the LLM response into GeneratedQAPair objects.

        Args:
            response: The raw LLM response string.

        Returns:
            List of GeneratedQAPair objects.
        """
        # Clean up response - remove markdown code blocks if present
        cleaned = response.strip()
        if cleaned.startswith("```"):
            # Remove opening code block
            lines = cleaned.split("\n")
            start_idx = 1 if lines[0].startswith("```") else 0
            end_idx = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == "```":
                    end_idx = i
                    break
            cleaned = "\n".join(lines[start_idx:end_idx])

        try:
            data = json.loads(cleaned)
            if not isinstance(data, list):
                logger.error("LLM response is not a JSON array")
                return []

            pairs: list[GeneratedQAPair] = []
            for item in data:
                if isinstance(item, dict) and "query" in item and "relevant_chunk_ids" in item:
                    pairs.append(
                        GeneratedQAPair(
                            query=item["query"],
                            relevant_chunk_ids=item["relevant_chunk_ids"],
                        )
                    )
            return pairs

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return []

    def _validate_chunk_ids(self, chunk_ids: list[str]) -> list[ChunkId]:
        """Validate that chunk IDs exist in the index.

        Args:
            chunk_ids: List of chunk ID strings to validate.

        Returns:
            List of valid ChunkId objects that exist in the index.
        """
        valid_ids: list[ChunkId] = []
        for cid in chunk_ids:
            chunk_id = ChunkId(cid)
            if chunk_id in self._chunk_index:
                valid_ids.append(chunk_id)
            else:
                logger.warning(f"Invalid chunk ID referenced: {cid}")
        return valid_ids

    def _upload_to_langsmith(
        self,
        ground_truth: list[ChunkLevelGroundTruth],
        dataset_name: str | None,
    ) -> None:
        """Upload ground truth data to LangSmith.

        Args:
            ground_truth: List of ground truth objects to upload.
            dataset_name: Name for the dataset. If None, auto-generated.
        """
        # Import LangSmith client
        try:
            from langsmith import Client
        except ImportError:
            logger.error("langsmith package not installed. Install with: pip install langsmith")
            return

        client = Client()

        # Generate dataset name if not provided
        if dataset_name is None:
            import time

            timestamp = int(time.time())
            dataset_name = f"rag-eval-chunk-level-{timestamp}"

        # Create or get dataset
        try:
            dataset = client.create_dataset(
                dataset_name=dataset_name,
                description="Chunk-level ground truth for RAG evaluation",
            )
            logger.info(f"Created LangSmith dataset: {dataset_name}")
        except Exception:
            # Dataset might already exist
            dataset = client.read_dataset(dataset_name=dataset_name)
            logger.info(f"Using existing LangSmith dataset: {dataset_name}")

        # Upload examples
        for gt in ground_truth:
            source_docs = gt.query.metadata.get("source_doc", "")
            client.create_example(
                dataset_id=dataset.id,
                inputs={"query": gt.query.text},
                outputs={"relevant_chunk_ids": list(gt.relevant_chunk_ids)},
                metadata={
                    "source_docs": [source_docs] if source_docs else [],
                    "generation_type": "synthetic",
                    "chunker": self._chunker.name,
                },
            )

        logger.info(f"Uploaded {len(ground_truth)} examples to LangSmith")


__all__ = [
    "ChunkLevelSyntheticDatasetGenerator",
    "GeneratedQAPair",
]
