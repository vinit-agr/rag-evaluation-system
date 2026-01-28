"""Tests for TokenLevelSyntheticDatasetGenerator."""

import json

import pytest
from pydantic import ValidationError

from rag_evaluation_system.synthetic_datagen.token_level.generator import (
    ExtractedExcerpt,
    GeneratedQAWithExcerpts,
    TokenLevelSyntheticDatasetGenerator,
)
from rag_evaluation_system.types import Corpus, Document
from tests.conftest import MockLLMClient


class TestTokenLevelSyntheticDatasetGeneratorInit:
    """Tests for TokenLevelSyntheticDatasetGenerator initialization."""

    def test_init_with_corpus_no_chunker_needed(
        self,
        sample_corpus: Corpus,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that generator initializes with corpus only (no chunker required)."""
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
        )

        assert generator.corpus == sample_corpus
        # Should not have any chunker attribute

    def test_init_stores_llm_client(
        self,
        sample_corpus: Corpus,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that LLM client is stored properly."""
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
        )

        assert generator._llm == mock_llm_client


class TestFindSpanPositions:
    """Tests for _find_span_positions method."""

    def test_find_span_positions_finds_exact_matches(
        self,
        sample_document: Document,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _find_span_positions finds exact text matches."""
        corpus = Corpus(documents=[sample_document])
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=corpus,
        )

        # Get an exact excerpt from the document
        excerpt = "Retrieval-Augmented Generation (RAG)"
        excerpts = [excerpt]

        spans = generator._find_span_positions(sample_document, excerpts)

        assert len(spans) == 1
        span = spans[0]
        assert span.doc_id == sample_document.id
        assert span.text == excerpt
        # Verify the position is correct
        assert sample_document.content[span.start : span.end] == excerpt

    def test_find_span_positions_handles_no_match(
        self,
        sample_document: Document,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _find_span_positions returns empty list when no match found."""
        corpus = Corpus(documents=[sample_document])
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=corpus,
        )

        # Use text that doesn't exist in the document
        excerpts = ["This text absolutely does not exist in the document XYZ123!@#"]

        spans = generator._find_span_positions(sample_document, excerpts)

        assert len(spans) == 0

    def test_find_span_positions_handles_multiple_excerpts(
        self,
        sample_document: Document,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _find_span_positions handles multiple excerpts."""
        corpus = Corpus(documents=[sample_document])
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=corpus,
        )

        excerpts = [
            "RAG systems reduce hallucination",
            "Document chunking and indexing",
        ]

        spans = generator._find_span_positions(sample_document, excerpts)

        assert len(spans) == 2
        for span in spans:
            assert span.doc_id == sample_document.id
            assert sample_document.content[span.start : span.end] == span.text

    def test_find_span_positions_partial_match(
        self,
        sample_document: Document,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _find_span_positions only returns found excerpts."""
        corpus = Corpus(documents=[sample_document])
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=corpus,
        )

        excerpts = [
            "RAG systems reduce hallucination",  # Exists
            "This text does not exist at all",  # Does not exist
        ]

        spans = generator._find_span_positions(sample_document, excerpts)

        # Should only find the first excerpt
        assert len(spans) == 1
        assert "hallucination" in spans[0].text


class TestFuzzyFind:
    """Tests for _fuzzy_find method."""

    def test_fuzzy_find_finds_similar_text(
        self,
        sample_document: Document,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _fuzzy_find finds text with minor differences."""
        corpus = Corpus(documents=[sample_document])
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=corpus,
        )

        # Slightly modified version (LLM might add/remove punctuation)
        # Original text: "Retrieval-Augmented Generation (RAG)"
        modified = "Retrieval-Augmented Generation  (RAG)"  # Extra space

        # Find with high threshold
        position = generator._fuzzy_find(sample_document.content, modified, threshold=0.9)

        # Should find a match (position >= 0)
        assert position >= 0

    def test_fuzzy_find_returns_negative_for_no_match(
        self,
        sample_document: Document,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _fuzzy_find returns -1 when no match found."""
        corpus = Corpus(documents=[sample_document])
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=corpus,
        )

        # Completely different text
        no_match = "Quantum computing in distributed systems"

        position = generator._fuzzy_find(sample_document.content, no_match, threshold=0.9)

        assert position == -1

    def test_fuzzy_find_handles_empty_excerpt(
        self,
        sample_document: Document,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _fuzzy_find handles empty excerpt."""
        corpus = Corpus(documents=[sample_document])
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=corpus,
        )

        position = generator._fuzzy_find(sample_document.content, "", threshold=0.9)

        assert position == -1

    def test_fuzzy_find_handles_excerpt_longer_than_text(
        self,
        sample_document: Document,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _fuzzy_find handles excerpt longer than text."""
        corpus = Corpus(documents=[sample_document])
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=corpus,
        )

        # Excerpt longer than the document
        long_excerpt = sample_document.content + " extra content that makes it longer"

        position = generator._fuzzy_find(sample_document.content, long_excerpt, threshold=0.9)

        assert position == -1


class TestParseQuestionsResponse:
    """Tests for _parse_questions_response method."""

    def test_parse_questions_response_handles_valid_json(
        self,
        sample_corpus: Corpus,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _parse_questions_response correctly parses valid JSON array."""
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
        )

        valid_json = json.dumps(
            [
                "What is RAG?",
                "How does retrieval work?",
                "What are the benefits?",
            ]
        )

        questions = generator._parse_questions_response(valid_json)

        assert len(questions) == 3
        assert questions[0] == "What is RAG?"
        assert questions[1] == "How does retrieval work?"
        assert questions[2] == "What are the benefits?"

    def test_parse_questions_response_handles_invalid_json(
        self,
        sample_corpus: Corpus,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _parse_questions_response returns empty list for invalid JSON."""
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
        )

        invalid_json = "Not valid JSON at all"

        questions = generator._parse_questions_response(invalid_json)

        assert questions == []

    def test_parse_questions_response_handles_non_array(
        self,
        sample_corpus: Corpus,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _parse_questions_response returns empty list for non-array JSON."""
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
        )

        non_array_json = '{"question": "What is RAG?"}'

        questions = generator._parse_questions_response(non_array_json)

        assert questions == []

    def test_parse_questions_response_filters_non_strings(
        self,
        sample_corpus: Corpus,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _parse_questions_response only includes string items."""
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
        )

        # Mix of strings and non-strings
        mixed_json = json.dumps(
            [
                "Valid question 1",
                123,  # Not a string
                "Valid question 2",
                {"nested": "object"},  # Not a string
                "Valid question 3",
            ]
        )

        questions = generator._parse_questions_response(mixed_json)

        # Should only have the 3 valid strings
        assert len(questions) == 3
        assert all(q.startswith("Valid question") for q in questions)


class TestParseExcerptsResponse:
    """Tests for _parse_excerpts_response method."""

    def test_parse_excerpts_response_handles_valid_json(
        self,
        sample_corpus: Corpus,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _parse_excerpts_response correctly parses valid JSON array."""
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
        )

        valid_json = json.dumps(
            [
                "RAG combines retrieval with generation.",
                "The retrieval component fetches relevant documents.",
            ]
        )

        excerpts = generator._parse_excerpts_response(valid_json)

        assert len(excerpts) == 2
        assert excerpts[0] == "RAG combines retrieval with generation."
        assert excerpts[1] == "The retrieval component fetches relevant documents."

    def test_parse_excerpts_response_handles_invalid_json(
        self,
        sample_corpus: Corpus,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _parse_excerpts_response returns empty list for invalid JSON."""
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
        )

        invalid_json = "This is not valid JSON"

        excerpts = generator._parse_excerpts_response(invalid_json)

        assert excerpts == []


class TestCleanJsonResponse:
    """Tests for _clean_json_response method."""

    def test_clean_json_response_removes_markdown_code_blocks(
        self,
        sample_corpus: Corpus,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _clean_json_response removes markdown code block formatting."""
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
        )

        with_code_blocks = """```json
["Question 1", "Question 2"]
```"""

        cleaned = generator._clean_json_response(with_code_blocks)

        # Should be valid JSON after cleaning
        result = json.loads(cleaned)
        assert result == ["Question 1", "Question 2"]

    def test_clean_json_response_removes_generic_code_blocks(
        self,
        sample_corpus: Corpus,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _clean_json_response removes generic code blocks."""
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
        )

        with_generic_blocks = """```
["Question 1"]
```"""

        cleaned = generator._clean_json_response(with_generic_blocks)

        result = json.loads(cleaned)
        assert result == ["Question 1"]

    def test_clean_json_response_preserves_clean_json(
        self,
        sample_corpus: Corpus,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _clean_json_response preserves already clean JSON."""
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
        )

        clean_json = '["Question 1", "Question 2"]'

        cleaned = generator._clean_json_response(clean_json)

        assert json.loads(cleaned) == json.loads(clean_json)

    def test_clean_json_response_handles_whitespace(
        self,
        sample_corpus: Corpus,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _clean_json_response handles leading/trailing whitespace."""
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
        )

        with_whitespace = '   \n["Question"]\n   '

        cleaned = generator._clean_json_response(with_whitespace)

        assert json.loads(cleaned) == ["Question"]


class TestModels:
    """Tests for Pydantic models used in token-level generation."""

    def test_extracted_excerpt_creation(self) -> None:
        """Test that ExtractedExcerpt can be created with valid data."""
        excerpt = ExtractedExcerpt(text="This is an excerpt from the document.")

        assert excerpt.text == "This is an excerpt from the document."

    def test_extracted_excerpt_is_frozen(self) -> None:
        """Test that ExtractedExcerpt is immutable."""
        excerpt = ExtractedExcerpt(text="Test excerpt")

        with pytest.raises(ValidationError):  # Pydantic frozen model raises ValidationError
            excerpt.text = "New text"  # type: ignore[misc]

    def test_generated_qa_with_excerpts_creation(self) -> None:
        """Test that GeneratedQAWithExcerpts can be created with valid data."""
        qa = GeneratedQAWithExcerpts(
            query="What is RAG?",
            excerpts=["Excerpt 1", "Excerpt 2"],
        )

        assert qa.query == "What is RAG?"
        assert qa.excerpts == ["Excerpt 1", "Excerpt 2"]

    def test_generated_qa_with_excerpts_is_frozen(self) -> None:
        """Test that GeneratedQAWithExcerpts is immutable."""
        qa = GeneratedQAWithExcerpts(
            query="Test query",
            excerpts=["Excerpt"],
        )

        with pytest.raises(ValidationError):  # Pydantic frozen model raises ValidationError
            qa.query = "New query"  # type: ignore[misc]


class TestGenerateWithMockLLM:
    """Tests for generate method using mock LLM."""

    def test_generate_calls_llm_for_questions_and_excerpts(
        self,
        sample_corpus: Corpus,
    ) -> None:
        """Test that generate calls LLM for both questions and excerpt extraction."""
        # For each document, we need:
        # 1. Questions response
        # 2. Excerpts response for each question

        # First document: 1 question -> 1 excerpt call
        # Second document: 1 question -> 1 excerpt call
        # Total: 2 question calls + 2 excerpt calls = 4 calls

        # Get exact text from documents for excerpts
        doc1_excerpt = sample_corpus.documents[0].content[31:75]  # Get an actual substring
        doc2_excerpt = sample_corpus.documents[1].content[20:60]  # Get an actual substring

        responses = [
            # Doc 1: questions
            '["What is RAG?"]',
            # Doc 1: excerpts for "What is RAG?"
            json.dumps([doc1_excerpt]),
            # Doc 2: questions
            '["What are vector databases?"]',
            # Doc 2: excerpts for "What are vector databases?"
            json.dumps([doc2_excerpt]),
        ]

        mock_llm = MockLLMClient(responses=responses)

        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm,
            corpus=sample_corpus,
        )

        generator.generate(queries_per_doc=1, upload_to_langsmith=False)

        # Should have made 4 calls (2 for questions, 2 for excerpts)
        assert len(mock_llm.call_history) == 4

    def test_generate_creates_ground_truth_with_character_spans(
        self,
        sample_document: Document,
    ) -> None:
        """Test that generate creates ground truth with proper character spans."""
        corpus = Corpus(documents=[sample_document])

        # Use exact text from document
        exact_excerpt = "Retrieval-Augmented Generation (RAG)"

        responses = [
            '["What does RAG stand for?"]',
            json.dumps([exact_excerpt]),
        ]

        mock_llm = MockLLMClient(responses=responses)

        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm,
            corpus=corpus,
        )

        ground_truth = generator.generate(queries_per_doc=1, upload_to_langsmith=False)

        assert len(ground_truth) == 1
        gt = ground_truth[0]

        assert gt.query.text == "What does RAG stand for?"
        assert len(gt.relevant_spans) == 1

        span = gt.relevant_spans[0]
        assert span.doc_id == sample_document.id
        assert span.text == exact_excerpt
        assert sample_document.content[span.start : span.end] == exact_excerpt

    def test_generate_skips_questions_without_valid_excerpts(
        self,
        sample_document: Document,
    ) -> None:
        """Test that generate skips questions when no valid excerpts are found."""
        corpus = Corpus(documents=[sample_document])

        responses = [
            '["What is mentioned?"]',
            # Excerpt that doesn't exist in document
            '["This text absolutely does not exist in the document XYZ123"]',
        ]

        mock_llm = MockLLMClient(responses=responses)

        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm,
            corpus=corpus,
        )

        ground_truth = generator.generate(queries_per_doc=1, upload_to_langsmith=False)

        # Should have no ground truth since excerpt wasn't found
        assert len(ground_truth) == 0

    def test_generate_query_ids_are_content_hashed(
        self,
        sample_document: Document,
    ) -> None:
        """Test that generated query IDs are based on content hash."""
        corpus = Corpus(documents=[sample_document])

        # Use exact text from document
        exact_excerpt = "Retrieval-Augmented Generation (RAG)"

        responses = [
            '["What does RAG stand for?"]',
            json.dumps([exact_excerpt]),
        ]

        mock_llm = MockLLMClient(responses=responses)

        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm,
            corpus=corpus,
        )

        ground_truth = generator.generate(queries_per_doc=1, upload_to_langsmith=False)

        # Query ID should start with "query_"
        assert len(ground_truth) > 0
        for gt in ground_truth:
            assert gt.query.id.startswith("query_")
            # ID should be deterministic (hash-based)
            assert len(gt.query.id) > len("query_")

    def test_generate_stores_source_doc_in_metadata(
        self,
        sample_document: Document,
    ) -> None:
        """Test that generated queries store source document ID in metadata."""
        corpus = Corpus(documents=[sample_document])

        # Use exact text from document
        exact_excerpt = "Retrieval-Augmented Generation (RAG)"

        responses = [
            '["What does RAG stand for?"]',
            json.dumps([exact_excerpt]),
        ]

        mock_llm = MockLLMClient(responses=responses)

        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm,
            corpus=corpus,
        )

        ground_truth = generator.generate(queries_per_doc=1, upload_to_langsmith=False)

        # Check that source_doc is in metadata
        assert len(ground_truth) > 0
        for gt in ground_truth:
            assert "source_doc" in gt.query.metadata

    def test_generate_handles_empty_questions_response(
        self,
        sample_document: Document,
    ) -> None:
        """Test that generate handles LLM returning no questions."""
        corpus = Corpus(documents=[sample_document])

        responses = [
            "[]",  # Empty questions array
        ]

        mock_llm = MockLLMClient(responses=responses)

        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm,
            corpus=corpus,
        )

        ground_truth = generator.generate(queries_per_doc=5, upload_to_langsmith=False)

        # Should have no ground truth since no questions were generated
        assert len(ground_truth) == 0

    def test_generate_handles_empty_excerpts_response(
        self,
        sample_document: Document,
    ) -> None:
        """Test that generate handles LLM returning no excerpts."""
        corpus = Corpus(documents=[sample_document])

        responses = [
            '["What is RAG?"]',
            "[]",  # Empty excerpts array
        ]

        mock_llm = MockLLMClient(responses=responses)

        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm,
            corpus=corpus,
        )

        ground_truth = generator.generate(queries_per_doc=1, upload_to_langsmith=False)

        # Should have no ground truth since no excerpts were returned
        assert len(ground_truth) == 0


class TestGenerateQuestions:
    """Tests for _generate_questions method."""

    def test_generate_questions_calls_llm(
        self,
        sample_document: Document,
    ) -> None:
        """Test that _generate_questions calls the LLM with correct prompts."""
        corpus = Corpus(documents=[sample_document])

        mock_llm = MockLLMClient(responses=['["Question 1?", "Question 2?"]'])

        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm,
            corpus=corpus,
        )

        questions = generator._generate_questions(sample_document, num_queries=2)

        assert len(questions) == 2
        assert questions[0] == "Question 1?"
        assert questions[1] == "Question 2?"
        assert len(mock_llm.call_history) == 1

    def test_generate_questions_includes_document_content(
        self,
        sample_document: Document,
    ) -> None:
        """Test that _generate_questions includes document content in prompt."""
        corpus = Corpus(documents=[sample_document])

        mock_llm = MockLLMClient(responses=['["Question?"]'])

        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm,
            corpus=corpus,
        )

        generator._generate_questions(sample_document, num_queries=1)

        # Check that document content was included in the prompt
        assert len(mock_llm.call_history) == 1
        user_message = next(
            msg for msg in mock_llm.call_history[0]["messages"] if msg["role"] == "user"
        )
        assert "Document:" in user_message["content"]


class TestExtractExcerpts:
    """Tests for _extract_excerpts method."""

    def test_extract_excerpts_calls_llm(
        self,
        sample_document: Document,
    ) -> None:
        """Test that _extract_excerpts calls the LLM."""
        corpus = Corpus(documents=[sample_document])

        mock_llm = MockLLMClient(responses=['["Excerpt text"]'])

        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm,
            corpus=corpus,
        )

        excerpts = generator._extract_excerpts(sample_document, "What is RAG?")

        assert len(excerpts) == 1
        assert excerpts[0] == "Excerpt text"
        assert len(mock_llm.call_history) == 1

    def test_extract_excerpts_includes_question(
        self,
        sample_document: Document,
    ) -> None:
        """Test that _extract_excerpts includes the question in prompt."""
        corpus = Corpus(documents=[sample_document])

        mock_llm = MockLLMClient(responses=['["Excerpt"]'])

        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm,
            corpus=corpus,
        )

        question = "What does RAG mean?"
        generator._extract_excerpts(sample_document, question)

        # Check that question was included in the prompt
        assert len(mock_llm.call_history) == 1
        user_message = next(
            msg for msg in mock_llm.call_history[0]["messages"] if msg["role"] == "user"
        )
        assert question in user_message["content"]


class TestCallLLM:
    """Tests for _call_llm method in token-level generator."""

    def test_call_llm_includes_system_prompt(
        self,
        sample_corpus: Corpus,
    ) -> None:
        """Test that _call_llm includes the system prompt."""
        mock_llm = MockLLMClient(responses=["test response"])

        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm,
            corpus=sample_corpus,
        )

        generator._call_llm("System prompt", "User prompt")

        # Check that both prompts were included
        assert len(mock_llm.call_history) == 1
        messages = mock_llm.call_history[0]["messages"]
        assert any(msg["role"] == "system" for msg in messages)
        assert any(msg["role"] == "user" for msg in messages)

    def test_call_llm_returns_content(
        self,
        sample_corpus: Corpus,
    ) -> None:
        """Test that _call_llm returns the response content."""
        mock_llm = MockLLMClient(responses=["expected response"])

        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm,
            corpus=sample_corpus,
        )

        result = generator._call_llm("System prompt", "User prompt")

        assert result == "expected response"


class TestFuzzyFindEdgeCases:
    """Additional edge case tests for _fuzzy_find method."""

    def test_fuzzy_find_with_lower_threshold(
        self,
        sample_document: Document,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _fuzzy_find works with lower similarity threshold."""
        corpus = Corpus(documents=[sample_document])
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=corpus,
        )

        # More significant modification from original: "Retrieval-Augmented Generation (RAG)"
        modified = "Retrieval Augmented Generation RAG"  # No hyphens, no parens

        # Should find with lower threshold
        position = generator._fuzzy_find(sample_document.content, modified, threshold=0.7)

        # With a lower threshold, should find a match
        assert position >= 0

    def test_fuzzy_find_exact_match_returns_position(
        self,
        sample_document: Document,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _fuzzy_find works for exact matches."""
        corpus = Corpus(documents=[sample_document])
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=corpus,
        )

        # Exact text from document
        exact_text = "RAG systems reduce hallucination"

        position = generator._fuzzy_find(sample_document.content, exact_text, threshold=1.0)

        # Should find exact match
        assert position >= 0
        assert sample_document.content[position : position + len(exact_text)] == exact_text


class TestParseResponsesEdgeCases:
    """Edge case tests for parsing responses."""

    def test_parse_questions_response_empty_array(
        self,
        sample_corpus: Corpus,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _parse_questions_response handles empty array."""
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
        )

        questions = generator._parse_questions_response("[]")

        assert questions == []

    def test_parse_excerpts_response_empty_array(
        self,
        sample_corpus: Corpus,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _parse_excerpts_response handles empty array."""
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
        )

        excerpts = generator._parse_excerpts_response("[]")

        assert excerpts == []

    def test_parse_excerpts_response_filters_non_strings(
        self,
        sample_corpus: Corpus,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _parse_excerpts_response filters non-string items."""
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
        )

        mixed_json = json.dumps(
            [
                "Valid excerpt",
                123,  # Not a string
                None,  # Not a string
                "Another valid excerpt",
            ]
        )

        excerpts = generator._parse_excerpts_response(mixed_json)

        assert len(excerpts) == 2
        assert excerpts[0] == "Valid excerpt"
        assert excerpts[1] == "Another valid excerpt"

    def test_clean_json_response_handles_empty_string(
        self,
        sample_corpus: Corpus,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _clean_json_response handles empty string."""
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
        )

        cleaned = generator._clean_json_response("")

        assert cleaned == ""

    def test_clean_json_response_handles_code_block_only(
        self,
        sample_corpus: Corpus,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _clean_json_response handles code block markers only."""
        generator = TokenLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
        )

        # Just code block markers
        cleaned = generator._clean_json_response("```\n```")

        # Should return empty after stripping markers
        assert cleaned == ""
