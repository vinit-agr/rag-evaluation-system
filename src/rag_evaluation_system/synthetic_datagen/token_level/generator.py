"""Token-level synthetic data generator."""
import json
import logging
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel

from rag_evaluation_system.types import (
    CharacterSpan,
    Corpus,
    Document,
    Query,
    QueryId,
    QueryText,
    TokenLevelGroundTruth,
)
from ..base import SyntheticDatasetGenerator

logger = logging.getLogger(__name__)


class ExtractedExcerpt(BaseModel):
    """An excerpt extracted by the LLM."""

    text: str


class GeneratedQAWithExcerpts(BaseModel):
    """A generated query with relevant excerpts."""

    query: str
    excerpts: list[str]


class TokenLevelSyntheticDatasetGenerator(SyntheticDatasetGenerator):
    """Generate synthetic QA pairs with character span ground truth."""

    QUERY_GENERATION_PROMPT = """You are an expert at generating evaluation questions.
Given a document, generate diverse questions that can be answered using
specific passages from the document.

Output JSON format:
{
    "questions": ["What is...?", "How does...?", ...]
}"""

    EXCERPT_EXTRACTION_PROMPT = """You are an expert at identifying relevant text.
Given a document and a question, extract the exact passages that answer
the question. Copy the text VERBATIM - do not paraphrase or summarize.

Output JSON format:
{
    "excerpts": ["exact text from document...", ...]
}"""

    def generate(
        self,
        queries_per_doc: int = 5,
        upload_to_langsmith: bool = True,
        dataset_name: str | None = None,
    ) -> list[TokenLevelGroundTruth]:
        ground_truth: list[TokenLevelGroundTruth] = []
        query_counter = 0

        for doc in self._corpus.documents:
            logger.info("Processing document: %s", doc.id)
            questions = self._generate_questions(doc, queries_per_doc)

            for question in questions:
                excerpts = self._extract_excerpts(doc, question)
                spans = self._find_span_positions(doc, excerpts)

                if not spans:
                    logger.warning("No spans found for: %s...", question[:50])
                    continue

                ground_truth.append(
                    TokenLevelGroundTruth(
                        query=Query(
                            id=QueryId(f"q_{query_counter}"),
                            text=QueryText(question),
                            metadata={"source_doc": str(doc.id)},
                        ),
                        relevant_spans=spans,
                    )
                )
                query_counter += 1

        if upload_to_langsmith:
            self._upload_to_langsmith(ground_truth, dataset_name)

        return ground_truth

    def _generate_questions(self, doc: Document, num_queries: int) -> list[str]:
        prompt = f"""Document:
{doc.content[:8000]}

Generate {num_queries} diverse questions about this document."""

        response = self._call_llm(self.QUERY_GENERATION_PROMPT, prompt)
        data = json.loads(response)
        return data.get("questions", [])

    def _extract_excerpts(self, doc: Document, question: str) -> list[str]:
        prompt = f"""Document:
{doc.content[:8000]}

Question: {question}

Extract the exact passages that answer this question. Copy verbatim."""

        response = self._call_llm(self.EXCERPT_EXTRACTION_PROMPT, prompt)
        data = json.loads(response)
        return data.get("excerpts", [])

    def _find_span_positions(self, doc: Document, excerpts: list[str]) -> list[CharacterSpan]:
        spans: list[CharacterSpan] = []

        for excerpt in excerpts:
            start = doc.content.find(excerpt)

            if start == -1:
                start = self._fuzzy_find(doc.content, excerpt)

            if start == -1:
                logger.warning(
                    "Could not locate excerpt in document %s: %s...",
                    doc.id,
                    excerpt[:50],
                )
                continue

            end = start + len(excerpt)
            spans.append(
                CharacterSpan(
                    doc_id=doc.id,
                    start=start,
                    end=end,
                    text=doc.content[start:end],
                )
            )

        return spans

    def _fuzzy_find(self, text: str, excerpt: str, threshold: float = 0.9) -> int:
        if not excerpt:
            return -1

        window = len(excerpt)
        best_start = -1
        best_score = 0.0

        for i in range(0, max(1, len(text) - window + 1)):
            candidate = text[i : i + window]
            score = SequenceMatcher(None, candidate, excerpt).ratio()
            if score > best_score:
                best_score = score
                best_start = i
            if best_score >= threshold:
                break

        return best_start if best_score >= threshold else -1

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        response = self._llm.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _upload_to_langsmith(
        self,
        ground_truth: list[TokenLevelGroundTruth],
        dataset_name: str | None,
    ) -> None:
        from rag_evaluation_system.langsmith.upload import upload_token_level_dataset

        upload_token_level_dataset(ground_truth, dataset_name)
