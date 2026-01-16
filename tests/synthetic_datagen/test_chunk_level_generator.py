"""Tests for ChunkLevelSyntheticDatasetGenerator."""

import json

import pytest
from pydantic import ValidationError

from rag_evaluation_system.chunkers import RecursiveCharacterChunker
from rag_evaluation_system.synthetic_datagen.chunk_level.generator import (
    ChunkLevelSyntheticDatasetGenerator,
    GeneratedQAPair,
)
from rag_evaluation_system.types import Corpus, Document, DocumentId
from rag_evaluation_system.utils.hashing import generate_chunk_id
from tests.conftest import MockLLMClient


class TestChunkLevelSyntheticDatasetGeneratorInit:
    """Tests for ChunkLevelSyntheticDatasetGenerator initialization."""

    def test_init_with_corpus_and_chunker(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that generator initializes with corpus and chunker."""
        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
            chunker=sample_chunker,
        )

        assert generator.corpus == sample_corpus
        assert generator._chunker == sample_chunker
        assert generator._chunk_index == {}

    def test_init_stores_llm_client(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that LLM client is stored properly."""
        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
            chunker=sample_chunker,
        )

        assert generator._llm == mock_llm_client


class TestBuildChunkIndex:
    """Tests for _build_chunk_index method."""

    def test_build_chunk_index_creates_proper_index(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _build_chunk_index creates a proper index of chunk IDs to content."""
        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
            chunker=sample_chunker,
        )

        index = generator._build_chunk_index()

        # Verify index is populated
        assert len(index) > 0

        # Verify all keys are valid chunk IDs (start with "chunk_")
        for chunk_id in index:
            assert chunk_id.startswith("chunk_")

        # Verify chunk IDs are content-based hashes
        for chunk_id, content in index.items():
            expected_id = generate_chunk_id(content)
            assert chunk_id == expected_id

    def test_build_chunk_index_clears_previous_index(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _build_chunk_index clears any previous index."""
        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
            chunker=sample_chunker,
        )

        # Build index twice
        generator._build_chunk_index()
        first_count = len(generator._chunk_index)

        generator._build_chunk_index()
        second_count = len(generator._chunk_index)

        # Should have same count (not doubled)
        assert first_count == second_count

    def test_build_chunk_index_includes_all_documents(
        self,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _build_chunk_index processes all documents in corpus."""
        doc1 = Document(id=DocumentId("doc1.md"), content="Document one content here.")
        doc2 = Document(id=DocumentId("doc2.md"), content="Document two content here.")
        corpus = Corpus(documents=[doc1, doc2])

        chunker = RecursiveCharacterChunker(chunk_size=1000, chunk_overlap=0)

        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=corpus,
            chunker=chunker,
        )

        index = generator._build_chunk_index()

        # Should have chunks from both documents
        assert doc1.content in index.values()
        assert doc2.content in index.values()


class TestParseResponse:
    """Tests for _parse_response method."""

    def test_parse_response_handles_valid_json(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _parse_response correctly parses valid JSON."""
        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
            chunker=sample_chunker,
        )

        valid_json = json.dumps(
            [
                {"query": "What is RAG?", "relevant_chunk_ids": ["chunk_abc123"]},
                {
                    "query": "How does retrieval work?",
                    "relevant_chunk_ids": ["chunk_def456", "chunk_ghi789"],
                },
            ]
        )

        pairs = generator._parse_response(valid_json)

        assert len(pairs) == 2
        assert pairs[0].query == "What is RAG?"
        assert pairs[0].relevant_chunk_ids == ["chunk_abc123"]
        assert pairs[1].query == "How does retrieval work?"
        assert pairs[1].relevant_chunk_ids == ["chunk_def456", "chunk_ghi789"]

    def test_parse_response_handles_markdown_code_blocks(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _parse_response handles markdown code block formatting."""
        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
            chunker=sample_chunker,
        )

        # Response with markdown code blocks
        response_with_code_blocks = """```json
[
  {"query": "What is RAG?", "relevant_chunk_ids": ["chunk_abc123"]}
]
```"""

        pairs = generator._parse_response(response_with_code_blocks)

        assert len(pairs) == 1
        assert pairs[0].query == "What is RAG?"

    def test_parse_response_handles_invalid_json_gracefully(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _parse_response returns empty list for invalid JSON."""
        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
            chunker=sample_chunker,
        )

        invalid_json = "This is not valid JSON at all"

        pairs = generator._parse_response(invalid_json)

        assert pairs == []

    def test_parse_response_handles_non_array_json(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _parse_response returns empty list for non-array JSON."""
        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
            chunker=sample_chunker,
        )

        non_array_json = '{"query": "test", "relevant_chunk_ids": []}'

        pairs = generator._parse_response(non_array_json)

        assert pairs == []

    def test_parse_response_handles_missing_fields(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _parse_response skips items with missing required fields."""
        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
            chunker=sample_chunker,
        )

        # Mix of valid and invalid items
        mixed_json = json.dumps(
            [
                {"query": "Valid question", "relevant_chunk_ids": ["chunk_123"]},
                {"query": "Missing chunk_ids"},  # Missing relevant_chunk_ids
                {"relevant_chunk_ids": ["chunk_456"]},  # Missing query
                {"query": "Another valid", "relevant_chunk_ids": ["chunk_789"]},
            ]
        )

        pairs = generator._parse_response(mixed_json)

        # Should only have the 2 valid items
        assert len(pairs) == 2
        assert pairs[0].query == "Valid question"
        assert pairs[1].query == "Another valid"


class TestValidateChunkIds:
    """Tests for _validate_chunk_ids method."""

    def test_validate_chunk_ids_filters_invalid_ids(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _validate_chunk_ids only returns IDs that exist in the index."""
        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
            chunker=sample_chunker,
        )

        # Build the chunk index first
        generator._build_chunk_index()

        # Get one valid ID from the index
        valid_id = next(iter(generator._chunk_index.keys()))
        invalid_id = "chunk_nonexistent123"

        result = generator._validate_chunk_ids([valid_id, invalid_id])

        # Should only contain the valid ID
        assert len(result) == 1
        assert result[0] == valid_id

    def test_validate_chunk_ids_returns_empty_for_all_invalid(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _validate_chunk_ids returns empty list when all IDs are invalid."""
        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
            chunker=sample_chunker,
        )

        # Build the chunk index
        generator._build_chunk_index()

        # All invalid IDs
        invalid_ids = ["chunk_fake1", "chunk_fake2", "chunk_fake3"]

        result = generator._validate_chunk_ids(invalid_ids)

        assert result == []

    def test_validate_chunk_ids_preserves_order(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _validate_chunk_ids preserves the order of valid IDs."""
        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
            chunker=sample_chunker,
        )

        # Build the chunk index
        generator._build_chunk_index()

        # Get multiple valid IDs
        valid_ids = list(generator._chunk_index.keys())[:3]

        result = generator._validate_chunk_ids(valid_ids)

        # Should preserve order - compare as ChunkId objects
        from rag_evaluation_system.types import ChunkId

        assert result == [ChunkId(id) for id in valid_ids]


class TestGeneratedQAPair:
    """Tests for GeneratedQAPair model."""

    def test_generated_qa_pair_creation(self) -> None:
        """Test that GeneratedQAPair can be created with valid data."""
        pair = GeneratedQAPair(
            query="What is RAG?",
            relevant_chunk_ids=["chunk_abc123", "chunk_def456"],
        )

        assert pair.query == "What is RAG?"
        assert pair.relevant_chunk_ids == ["chunk_abc123", "chunk_def456"]

    def test_generated_qa_pair_is_frozen(self) -> None:
        """Test that GeneratedQAPair is immutable."""
        pair = GeneratedQAPair(
            query="What is RAG?",
            relevant_chunk_ids=["chunk_abc123"],
        )

        with pytest.raises(ValidationError):  # Pydantic frozen model raises ValidationError
            pair.query = "New query"  # type: ignore[misc]


class TestGenerateWithMockLLM:
    """Tests for generate method using mock LLM."""

    def test_generate_calls_llm_for_each_document(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
    ) -> None:
        """Test that generate calls LLM for each document in corpus."""
        # Create mock with responses for each document
        mock_llm = MockLLMClient(
            responses=[
                "[]",  # Response for doc 1
                "[]",  # Response for doc 2
            ]
        )

        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=mock_llm,
            corpus=sample_corpus,
            chunker=sample_chunker,
        )

        generator.generate(queries_per_doc=3, upload_to_langsmith=False)

        # Should have called LLM twice (once per document)
        assert len(mock_llm.call_history) == 2

    def test_generate_with_valid_qa_pairs(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
    ) -> None:
        """Test that generate creates ground truth from valid QA pairs."""
        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=MockLLMClient(),  # Temporary, we'll replace after getting chunk IDs
            corpus=sample_corpus,
            chunker=sample_chunker,
        )

        # Build chunk index to get valid IDs
        generator._build_chunk_index()
        valid_chunk_ids = list(generator._chunk_index.keys())[:2]

        # Create mock LLM with valid chunk ID references
        qa_response = json.dumps(
            [
                {
                    "query": "What is the purpose of RAG?",
                    "relevant_chunk_ids": valid_chunk_ids[:1],
                },
            ]
        )

        mock_llm = MockLLMClient(responses=[qa_response, qa_response])
        generator._llm = mock_llm

        ground_truth = generator.generate(queries_per_doc=1, upload_to_langsmith=False)

        # Should have generated ground truth
        assert len(ground_truth) > 0
        for gt in ground_truth:
            assert gt.query.text is not None
            assert len(gt.relevant_chunk_ids) > 0

    def test_generate_skips_documents_without_chunks(
        self,
    ) -> None:
        """Test that generate skips documents that produce no chunks."""
        # Create corpus with empty document
        empty_doc = Document(id=DocumentId("empty.md"), content="")
        corpus = Corpus(documents=[empty_doc])

        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=0)
        mock_llm = MockLLMClient()

        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=mock_llm,
            corpus=corpus,
            chunker=chunker,
        )

        ground_truth = generator.generate(queries_per_doc=3, upload_to_langsmith=False)

        # Should have no ground truth (empty doc produces no chunks)
        assert ground_truth == []
        # LLM should not have been called
        assert len(mock_llm.call_history) == 0

    def test_generate_skips_qa_pairs_with_no_valid_chunk_ids(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
    ) -> None:
        """Test that generate skips QA pairs with no valid chunk IDs."""
        # Response with invalid chunk IDs
        qa_response = json.dumps(
            [
                {
                    "query": "What is RAG?",
                    "relevant_chunk_ids": ["chunk_invalid123", "chunk_invalid456"],
                },
            ]
        )

        mock_llm = MockLLMClient(responses=[qa_response, qa_response])

        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=mock_llm,
            corpus=sample_corpus,
            chunker=sample_chunker,
        )

        ground_truth = generator.generate(queries_per_doc=1, upload_to_langsmith=False)

        # Should have no ground truth since all chunk IDs are invalid
        assert ground_truth == []

    def test_generate_query_ids_are_content_hashed(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
    ) -> None:
        """Test that generated query IDs are based on content hash."""
        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=MockLLMClient(),
            corpus=sample_corpus,
            chunker=sample_chunker,
        )

        # Build chunk index to get valid IDs
        generator._build_chunk_index()
        valid_chunk_ids = list(generator._chunk_index.keys())[:2]

        # Create mock LLM with deterministic response
        qa_response = json.dumps(
            [
                {
                    "query": "What is the purpose of RAG systems?",
                    "relevant_chunk_ids": valid_chunk_ids[:1],
                },
            ]
        )

        mock_llm = MockLLMClient(responses=[qa_response, qa_response])
        generator._llm = mock_llm

        ground_truth = generator.generate(queries_per_doc=1, upload_to_langsmith=False)

        # Query ID should start with "query_"
        assert len(ground_truth) > 0
        for gt in ground_truth:
            assert gt.query.id.startswith("query_")
            # ID should be deterministic (hash-based)
            assert len(gt.query.id) > len("query_")

    def test_generate_stores_source_doc_in_metadata(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
    ) -> None:
        """Test that generated queries store source document ID in metadata."""
        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=MockLLMClient(),
            corpus=sample_corpus,
            chunker=sample_chunker,
        )

        # Build chunk index to get valid IDs
        generator._build_chunk_index()
        valid_chunk_ids = list(generator._chunk_index.keys())[:2]

        # Create mock LLM response
        qa_response = json.dumps(
            [
                {
                    "query": "Test question?",
                    "relevant_chunk_ids": valid_chunk_ids[:1],
                },
            ]
        )

        mock_llm = MockLLMClient(responses=[qa_response, qa_response])
        generator._llm = mock_llm

        ground_truth = generator.generate(queries_per_doc=1, upload_to_langsmith=False)

        # Check that source_doc is in metadata
        assert len(ground_truth) > 0
        for gt in ground_truth:
            assert "source_doc" in gt.query.metadata


class TestGetDocumentChunks:
    """Tests for _get_document_chunks method."""

    def test_get_document_chunks_returns_chunks(
        self,
        sample_document: Document,
        sample_chunker: RecursiveCharacterChunker,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _get_document_chunks returns Chunk objects."""
        corpus = Corpus(documents=[sample_document])
        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=corpus,
            chunker=sample_chunker,
        )

        from rag_evaluation_system.types import Chunk

        chunks = generator._get_document_chunks(sample_document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.doc_id == sample_document.id
            assert chunk.content != ""

    def test_get_document_chunks_has_correct_chunk_ids(
        self,
        sample_document: Document,
        sample_chunker: RecursiveCharacterChunker,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that chunks have correct content-based IDs."""
        corpus = Corpus(documents=[sample_document])
        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=corpus,
            chunker=sample_chunker,
        )

        chunks = generator._get_document_chunks(sample_document)

        for chunk in chunks:
            expected_id = generate_chunk_id(chunk.content)
            assert chunk.id == expected_id

    def test_get_document_chunks_empty_document(
        self,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _get_document_chunks returns empty list for empty document."""
        empty_doc = Document(id=DocumentId("empty.md"), content="")
        corpus = Corpus(documents=[empty_doc])
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=0)

        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=corpus,
            chunker=chunker,
        )

        chunks = generator._get_document_chunks(empty_doc)

        assert chunks == []


class TestCallLLM:
    """Tests for _call_llm method."""

    def test_call_llm_includes_system_prompt(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
    ) -> None:
        """Test that _call_llm includes the system prompt."""
        mock_llm = MockLLMClient(responses=["test response"])

        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=mock_llm,
            corpus=sample_corpus,
            chunker=sample_chunker,
        )

        generator._call_llm("test user prompt")

        # Check that system prompt was included in the call
        assert len(mock_llm.call_history) == 1
        messages = mock_llm.call_history[0]["messages"]
        assert any(msg["role"] == "system" for msg in messages)
        assert any(msg["role"] == "user" for msg in messages)

    def test_call_llm_returns_content(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
    ) -> None:
        """Test that _call_llm returns the response content."""
        mock_llm = MockLLMClient(responses=["expected response"])

        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=mock_llm,
            corpus=sample_corpus,
            chunker=sample_chunker,
        )

        result = generator._call_llm("test prompt")

        assert result == "expected response"


class TestParseResponseEdgeCases:
    """Additional edge case tests for _parse_response method."""

    def test_parse_response_handles_empty_array(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _parse_response handles empty JSON array."""
        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
            chunker=sample_chunker,
        )

        pairs = generator._parse_response("[]")

        assert pairs == []

    def test_parse_response_handles_empty_string(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _parse_response handles empty string."""
        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
            chunker=sample_chunker,
        )

        pairs = generator._parse_response("")

        assert pairs == []

    def test_parse_response_handles_whitespace_only(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _parse_response handles whitespace-only string."""
        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
            chunker=sample_chunker,
        )

        pairs = generator._parse_response("   \n\t  ")

        assert pairs == []

    def test_parse_response_handles_nested_code_blocks(
        self,
        sample_corpus: Corpus,
        sample_chunker: RecursiveCharacterChunker,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Test that _parse_response handles code blocks with language specifier."""
        generator = ChunkLevelSyntheticDatasetGenerator(
            llm_client=mock_llm_client,
            corpus=sample_corpus,
            chunker=sample_chunker,
        )

        # Response with code block and language specifier
        response = """```json
[{"query": "Test?", "relevant_chunk_ids": ["chunk_123"]}]
```"""

        pairs = generator._parse_response(response)

        assert len(pairs) == 1
        assert pairs[0].query == "Test?"
