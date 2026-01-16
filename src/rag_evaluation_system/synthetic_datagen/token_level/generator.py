"""Token-level synthetic data generator.

This module implements synthetic QA pair generation with character span ground truth.
Unlike chunk-level generation, this does NOT require a chunker - ground truth is
expressed as raw character spans from the source documents.
"""

import hashlib
import json
import logging
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, ConfigDict

from rag_evaluation_system.synthetic_datagen.base import SyntheticDatasetGenerator
from rag_evaluation_system.types.chunks import CharacterSpan
from rag_evaluation_system.types.documents import Corpus, Document
from rag_evaluation_system.types.ground_truth import TokenLevelGroundTruth
from rag_evaluation_system.types.primitives import QueryId, QueryText
from rag_evaluation_system.types.queries import Query

logger = logging.getLogger(__name__)


class ExtractedExcerpt(BaseModel):
    """An excerpt extracted from a document by the LLM.

    Attributes:
        text: The verbatim text excerpt from the document.
    """

    model_config = ConfigDict(frozen=True)

    text: str


class GeneratedQAWithExcerpts(BaseModel):
    """A generated question with its relevant excerpts.

    Attributes:
        query: The generated question text.
        excerpts: List of verbatim text excerpts that answer the question.
    """

    model_config = ConfigDict(frozen=True)

    query: str
    excerpts: list[str]


class TokenLevelSyntheticDatasetGenerator(SyntheticDatasetGenerator):
    """Generate synthetic QA pairs with character span ground truth.

    This generator does NOT require a chunker. There is NO chunking at
    synthetic data generation time. Instead, it:
    1. Generates queries from document content
    2. Asks LLM to extract relevant excerpts (raw text)
    3. Finds character positions of excerpts in source document
    4. Stores as CharacterSpan objects (doc_id, start, end, text)

    This approach is chunker-independent, allowing fair comparison of
    different chunking strategies against the same ground truth.

    Example:
        >>> from openai import OpenAI
        >>> generator = TokenLevelSyntheticDatasetGenerator(
        ...     llm_client=OpenAI(),
        ...     corpus=corpus,
        ... )
        >>> ground_truth = generator.generate(queries_per_doc=5)
    """

    QUERY_GENERATION_PROMPT = """You are a helpful assistant that generates diverse questions about document content.

Given a document, generate questions that can be answered using information in the document.

Rules:
1. Generate diverse questions covering different aspects of the content
2. Questions should be answerable using ONLY the provided document
3. Include a mix of:
   - Factual questions (who, what, when, where)
   - Explanatory questions (how, why)
   - Comparative questions (differences, similarities)
   - Definition questions (what is X?)
4. Questions should be clear, specific, and unambiguous
5. Avoid trivial questions or questions requiring external knowledge

Output Format:
Return a JSON array of question strings.

Example output:
["What is the main purpose of RAG systems?", "How does retrieval augmentation help reduce hallucination?", "What are the key components of a RAG pipeline?"]"""

    EXCERPT_EXTRACTION_PROMPT = """You are a helpful assistant that extracts relevant passages from documents.

Given a document and a question, extract the exact passages that contain the answer.

Rules:
1. Copy text VERBATIM from the document - do not paraphrase or modify
2. Include enough context to make the excerpt self-contained
3. Extract multiple passages if the answer spans different parts
4. Each excerpt should be meaningful on its own
5. Include complete sentences when possible
6. Do NOT add any text that is not in the original document

Output Format:
Return a JSON array of excerpt strings, copied exactly from the document.

Example output:
["RAG combines retrieval with generation to provide more accurate responses.", "The retrieval component fetches relevant documents from a knowledge base before generation."]

IMPORTANT: Every character in your excerpts must appear exactly as written in the document."""

    def __init__(
        self,
        llm_client: Any,
        corpus: Corpus,
    ) -> None:
        """Initialize the token-level generator.

        Args:
            llm_client: An LLM client with an OpenAI-compatible interface.
            corpus: The document corpus to generate synthetic data from.

        Note:
            No chunker is required - ground truth is chunker-independent.
        """
        super().__init__(llm_client, corpus)

    def generate(
        self,
        queries_per_doc: int = 5,
        upload_to_langsmith: bool = True,
        dataset_name: str | None = None,
    ) -> list[TokenLevelGroundTruth]:
        """Generate synthetic queries with relevant character spans.

        Process:
        1. For each document:
           a. Ask LLM to generate queries about the document
           b. For each query, ask LLM to extract verbatim relevant excerpts
        2. For each excerpt:
           a. Find exact character positions in source document
           b. Create CharacterSpan with (doc_id, start, end, text)
        3. Upload to LangSmith if requested
        4. Return ground truth with CharacterSpan lists

        Args:
            queries_per_doc: Number of queries to generate per document.
            upload_to_langsmith: Whether to upload the dataset to LangSmith.
            dataset_name: Name for the LangSmith dataset. If None, auto-generated.

        Returns:
            List of TokenLevelGroundTruth objects containing queries and
            their relevant character spans.
        """
        ground_truth_list: list[TokenLevelGroundTruth] = []

        for doc in self._corpus.documents:
            logger.info(f"Processing document: {doc.id}")

            # Generate questions for this document
            questions = self._generate_questions(doc, queries_per_doc)

            for question in questions:
                # Extract relevant excerpts for this question
                excerpts = self._extract_excerpts(doc, question)

                if not excerpts:
                    logger.warning(f"No excerpts extracted for question: {question[:50]}...")
                    continue

                # Find character positions for excerpts
                spans = self._find_span_positions(doc, excerpts)

                if not spans:
                    logger.warning(
                        f"Could not locate any excerpts for question: {question[:50]}..."
                    )
                    continue

                # Generate query ID from content hash
                query_id = QueryId(f"query_{hashlib.sha256(question.encode()).hexdigest()[:12]}")

                ground_truth = TokenLevelGroundTruth(
                    query=Query(
                        id=query_id,
                        text=QueryText(question),
                        metadata={"source_doc": doc.id},
                    ),
                    relevant_spans=spans,
                )
                ground_truth_list.append(ground_truth)

        # Upload to LangSmith if requested
        if upload_to_langsmith and ground_truth_list:
            self._upload_to_langsmith(ground_truth_list, dataset_name)

        return ground_truth_list

    def _generate_questions(self, doc: Document, num_queries: int) -> list[str]:
        """Generate questions about a document.

        Args:
            doc: The document to generate questions about.
            num_queries: Number of questions to generate.

        Returns:
            List of question strings.
        """
        user_prompt = f"""Document:
{doc.content}

Generate exactly {num_queries} diverse questions that can be answered using this document.

Return ONLY a valid JSON array of question strings."""

        response = self._call_llm(self.QUERY_GENERATION_PROMPT, user_prompt)

        # Parse response
        questions = self._parse_questions_response(response)

        if len(questions) < num_queries:
            logger.warning(f"Generated {len(questions)} questions instead of {num_queries}")

        return questions

    def _extract_excerpts(self, doc: Document, question: str) -> list[str]:
        """Extract relevant excerpts from a document for a question.

        Args:
            doc: The source document.
            question: The question to find answers for.

        Returns:
            List of verbatim excerpt strings from the document.
        """
        user_prompt = f"""Document:
{doc.content}

Question: {question}

Extract the exact passages from the document that answer this question.
Copy the text VERBATIM - do not paraphrase.

Return ONLY a valid JSON array of excerpt strings."""

        response = self._call_llm(self.EXCERPT_EXTRACTION_PROMPT, user_prompt)

        # Parse response
        return self._parse_excerpts_response(response)

    def _find_span_positions(self, doc: Document, excerpts: list[str]) -> list[CharacterSpan]:
        """Find character positions of excerpts in the document.

        Args:
            doc: The source document.
            excerpts: List of excerpt strings to locate.

        Returns:
            List of CharacterSpan objects with positions found in the document.
        """
        spans: list[CharacterSpan] = []

        for excerpt in excerpts:
            # Try exact match first
            start = doc.content.find(excerpt)

            if start == -1:
                # Try fuzzy matching
                start = self._fuzzy_find(doc.content, excerpt)

            if start == -1:
                logger.warning(f"Could not locate excerpt in document: {excerpt[:50]}...")
                continue

            # Get the actual text from the document at this position
            end = start + len(excerpt)
            actual_text = doc.content[start:end]

            # If fuzzy match, use the actual document text
            if actual_text != excerpt:
                logger.debug(
                    f"Using fuzzy match: original='{excerpt[:30]}...' "
                    f"actual='{actual_text[:30]}...'"
                )

            span = CharacterSpan(
                doc_id=doc.id,
                start=start,
                end=end,
                text=actual_text,
            )
            spans.append(span)

        return spans

    def _fuzzy_find(self, text: str, excerpt: str, threshold: float = 0.9) -> int:
        """Find an excerpt in text using fuzzy matching.

        Uses sliding window with SequenceMatcher to find the best match
        above the similarity threshold.

        Args:
            text: The full text to search in.
            excerpt: The excerpt to find.
            threshold: Minimum similarity ratio (0.0 to 1.0) to accept a match.

        Returns:
            Starting position of the best match, or -1 if no match above threshold.
        """
        excerpt_len = len(excerpt)
        if excerpt_len == 0 or excerpt_len > len(text):
            return -1

        best_ratio = 0.0
        best_pos = -1

        # Slide a window of the excerpt length across the text
        # Use step size to balance accuracy vs performance
        step = max(1, excerpt_len // 10)

        for i in range(0, len(text) - excerpt_len + 1, step):
            window = text[i : i + excerpt_len]
            ratio = SequenceMatcher(None, excerpt, window).ratio()

            if ratio > best_ratio:
                best_ratio = ratio
                best_pos = i

        # Refine search around best position
        if best_pos > 0 and best_ratio > threshold * 0.9:
            search_start = max(0, best_pos - step)
            search_end = min(len(text) - excerpt_len + 1, best_pos + step)

            for i in range(search_start, search_end):
                window = text[i : i + excerpt_len]
                ratio = SequenceMatcher(None, excerpt, window).ratio()

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pos = i

        if best_ratio >= threshold:
            return best_pos

        return -1

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM with system and user prompts.

        Args:
            system_prompt: The system message to set context.
            user_prompt: The user message to send.

        Returns:
            The LLM's response text.
        """
        response = self._llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content or ""

    def _parse_questions_response(self, response: str) -> list[str]:
        """Parse LLM response into list of questions.

        Args:
            response: The raw LLM response string.

        Returns:
            List of question strings.
        """
        cleaned = self._clean_json_response(response)

        try:
            data = json.loads(cleaned)
            if not isinstance(data, list):
                logger.error("LLM response is not a JSON array")
                return []

            return [str(q) for q in data if isinstance(q, str)]

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse questions response as JSON: {e}")
            return []

    def _parse_excerpts_response(self, response: str) -> list[str]:
        """Parse LLM response into list of excerpts.

        Args:
            response: The raw LLM response string.

        Returns:
            List of excerpt strings.
        """
        cleaned = self._clean_json_response(response)

        try:
            data = json.loads(cleaned)
            if not isinstance(data, list):
                logger.error("LLM response is not a JSON array")
                return []

            return [str(e) for e in data if isinstance(e, str)]

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse excerpts response as JSON: {e}")
            return []

    def _clean_json_response(self, response: str) -> str:
        """Clean up LLM response by removing markdown code blocks.

        Args:
            response: The raw response string.

        Returns:
            Cleaned response string.
        """
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            start_idx = 1 if lines[0].startswith("```") else 0
            end_idx = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == "```":
                    end_idx = i
                    break
            cleaned = "\n".join(lines[start_idx:end_idx])
        return cleaned

    def _upload_to_langsmith(
        self,
        ground_truth: list[TokenLevelGroundTruth],
        dataset_name: str | None,
    ) -> None:
        """Upload ground truth data to LangSmith.

        Args:
            ground_truth: List of ground truth objects to upload.
            dataset_name: Name for the dataset. If None, auto-generated.
        """
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
            dataset_name = f"rag-eval-token-level-{timestamp}"

        # Create or get dataset
        try:
            dataset = client.create_dataset(
                dataset_name=dataset_name,
                description="Token-level ground truth for RAG evaluation (character spans)",
            )
            logger.info(f"Created LangSmith dataset: {dataset_name}")
        except Exception:
            dataset = client.read_dataset(dataset_name=dataset_name)
            logger.info(f"Using existing LangSmith dataset: {dataset_name}")

        # Upload examples
        for gt in ground_truth:
            source_docs = gt.query.metadata.get("source_doc", "")

            # Convert CharacterSpan objects to dictionaries
            spans_data = [
                {
                    "doc_id": span.doc_id,
                    "start": span.start,
                    "end": span.end,
                    "text": span.text,
                }
                for span in gt.relevant_spans
            ]

            client.create_example(
                dataset_id=dataset.id,
                inputs={"query": gt.query.text},
                outputs={"relevant_spans": spans_data},
                metadata={
                    "source_docs": [source_docs] if source_docs else [],
                    "generation_type": "synthetic",
                },
            )

        logger.info(f"Uploaded {len(ground_truth)} examples to LangSmith")


__all__ = [
    "ExtractedExcerpt",
    "GeneratedQAWithExcerpts",
    "TokenLevelSyntheticDatasetGenerator",
]
