"""Streamlit app for RAG Evaluation System synthetic data generation.

This app provides a user-friendly interface for generating synthetic evaluation
data for RAG systems in both chunk-level and token-level evaluation modes.
"""

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_suggested_folders() -> list[tuple[str, str]]:
    """Get a list of suggested folders that might contain markdown files.

    Returns:
        List of tuples (folder_path, display_label) for suggested folders.
    """
    suggestions: list[tuple[str, str]] = []

    # Check for data folders in the project
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"

    if data_dir.exists():
        # Look for help_hellotars_com specifically
        help_dir = data_dir / "help_hellotars_com"
        if help_dir.exists():
            # Add tiny_corpus
            tiny_corpus = help_dir / "tiny_corpus"
            if tiny_corpus.exists():
                suggestions.append((str(tiny_corpus), "Tiny Corpus"))

            # Add small_corpus
            small_corpus = help_dir / "small_corpus"
            if small_corpus.exists():
                suggestions.append((str(small_corpus), "Small Corpus"))

            # Add markdown as full corpus
            markdown_dir = help_dir / "markdown"
            if markdown_dir.exists():
                suggestions.append((str(markdown_dir), "Full Corpus"))

        # Check other data subfolders
        for subfolder in sorted(data_dir.iterdir()):
            if subfolder.is_dir() and subfolder.name != "help_hellotars_com":
                markdown_dir = subfolder / "markdown"
                if markdown_dir.exists():
                    suggestions.append((str(markdown_dir), f"{subfolder.name}"))

    return suggestions

if TYPE_CHECKING:
    from rag_evaluation_system.types import (
        CharacterSpan,
        ChunkLevelGroundTruth,
        Corpus,
        TokenLevelGroundTruth,
    )


def highlight_spans_in_document(content: str, spans: list[CharacterSpan]) -> str:
    """Highlight character spans in document content using HTML mark tags.

    Args:
        content: The full document content.
        spans: List of CharacterSpan objects to highlight.

    Returns:
        HTML string with <mark> tags around the relevant spans.
    """
    if not spans:
        return f"<pre>{html.escape(content)}</pre>"

    # Sort spans by start position
    sorted_spans = sorted(spans, key=lambda s: s.start)

    # Merge overlapping spans
    merged_spans: list[tuple[int, int]] = []
    for span in sorted_spans:
        if merged_spans and span.start <= merged_spans[-1][1]:
            # Overlapping with previous span, extend it
            merged_spans[-1] = (merged_spans[-1][0], max(merged_spans[-1][1], span.end))
        else:
            merged_spans.append((span.start, span.end))

    # Build HTML with highlights
    result_parts: list[str] = []
    last_end = 0

    for start, end in merged_spans:
        # Add text before this span
        if start > last_end:
            result_parts.append(html.escape(content[last_end:start]))
        # Add highlighted span
        result_parts.append(
            f'<mark style="background-color: #ffeb3b; padding: 2px;">'
            f"{html.escape(content[start:end])}</mark>"
        )
        last_end = end

    # Add remaining text after last span
    if last_end < len(content):
        result_parts.append(html.escape(content[last_end:]))

    return (
        f'<pre style="white-space: pre-wrap; word-wrap: break-word;">{"".join(result_parts)}</pre>'
    )


def load_corpus(folder_path: str) -> Corpus | None:
    """Load corpus from folder path.

    Args:
        folder_path: Path to folder containing markdown files.

    Returns:
        Corpus object or None if loading fails.
    """
    from rag_evaluation_system.types import Corpus

    try:
        path = Path(folder_path)
        if not path.exists():
            st.error(f"Folder not found: {folder_path}")
            return None
        if not path.is_dir():
            st.error(f"Path is not a directory: {folder_path}")
            return None

        corpus = Corpus.from_folder(folder_path, glob_pattern="**/*.md")

        if len(corpus.documents) == 0:
            st.warning("No markdown files found in the specified folder.")
            return None

        return corpus
    except Exception as e:
        st.error(f"Error loading corpus: {e}")
        return None


def check_stop_requested() -> bool:
    """Check if stop has been requested via session state.

    Returns:
        True if stop was requested, False otherwise.
    """
    return st.session_state.get("stop_generation", False)


def set_stop_flag() -> None:
    """Callback to set the stop generation flag."""
    st.session_state.stop_generation = True


def generate_chunk_level_data_with_progress(
    corpus: Corpus,
    chunk_size: int,
    chunk_overlap: int,
    queries_per_doc: int,
    api_key: str,
    progress_container,
    questions_container,
) -> list[ChunkLevelGroundTruth] | None:
    """Generate chunk-level synthetic data with real-time progress updates.

    Args:
        corpus: The document corpus.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between chunks.
        queries_per_doc: Number of queries per document.
        api_key: OpenAI API key.
        progress_container: Streamlit container for progress bar.
        questions_container: Streamlit container for showing generated questions.

    Returns:
        List of ChunkLevelGroundTruth objects or None if generation fails/stopped.
    """
    import hashlib

    from openai import OpenAI

    from rag_evaluation_system.chunkers.recursive_character import RecursiveCharacterChunker
    from rag_evaluation_system.types import Query, QueryId, QueryText
    from rag_evaluation_system.types.ground_truth import ChunkLevelGroundTruth
    from rag_evaluation_system.utils.hashing import generate_chunk_id

    try:
        client = OpenAI(api_key=api_key)
        chunker = RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Build chunk index first
        chunk_index: dict[str, str] = {}
        for doc in corpus.documents:
            chunk_texts = chunker.chunk(doc.content)
            for chunk_text in chunk_texts:
                cid = generate_chunk_id(chunk_text)
                chunk_index[cid] = chunk_text

        all_results: list[ChunkLevelGroundTruth] = []
        total_docs = len(corpus.documents)

        # Progress bar
        progress_bar = progress_container.progress(0, text="Starting generation...")
        status_text = progress_container.empty()

        for doc_idx, doc in enumerate(corpus.documents):
            # Check for stop request
            if check_stop_requested():
                progress_bar.progress(1.0, text="Generation stopped by user")
                status_text.write(f"**Stopped** - Generated {len(all_results)} questions")
                return all_results if all_results else None

            # Update progress
            progress = (doc_idx) / total_docs
            progress_bar.progress(progress, text=f"Processing document {doc_idx + 1}/{total_docs}")
            status_text.write(f"**Document:** {doc.id}")

            # Get chunks for this document
            doc_chunks = chunker.chunk(doc.content)
            if not doc_chunks:
                continue

            # Format chunks for prompt
            chunks_with_ids = []
            for chunk_text in doc_chunks:
                cid = generate_chunk_id(chunk_text)
                chunks_with_ids.append((cid, chunk_text))

            chunks_text = "\n\n".join(
                f"[{cid}]: {content}" for cid, content in chunks_with_ids[:20]
            )

            # Call LLM
            system_prompt = """You are a helpful assistant that generates question-answer pairs for evaluating retrieval systems.

Given a set of text chunks with their IDs, generate questions that can be answered using the information in specific chunks.

Rules:
1. Generate diverse questions that cover different aspects of the content
2. Questions should be answerable using ONLY the provided chunks
3. Each question should cite the specific chunk IDs that contain the answer
4. Questions should be clear, specific, and unambiguous

Output Format:
Return a JSON array of objects, each with:
- "query": The question text
- "relevant_chunk_ids": Array of chunk IDs that contain the answer

Important: Only use chunk IDs that were provided to you. Do not invent chunk IDs."""

            user_prompt = f"""Here are the text chunks with their IDs:

{chunks_text}

Generate exactly {queries_per_doc} diverse questions that can be answered using these chunks.
Return ONLY a valid JSON array."""

            import json

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
            )

            response_text = response.choices[0].message.content or ""

            # Parse response
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
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
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "query" in item and "relevant_chunk_ids" in item:
                            # Validate chunk IDs
                            valid_ids = [
                                cid for cid in item["relevant_chunk_ids"]
                                if cid in chunk_index
                            ]
                            if valid_ids:
                                query_id = QueryId(
                                    f"query_{hashlib.sha256(item['query'].encode()).hexdigest()[:12]}"
                                )
                                gt = ChunkLevelGroundTruth(
                                    query=Query(
                                        id=query_id,
                                        text=QueryText(item["query"]),
                                        metadata={"source_doc": doc.id},
                                    ),
                                    relevant_chunk_ids=valid_ids,
                                )
                                all_results.append(gt)

                                # Update questions display in real-time
                                with questions_container:
                                    st.markdown(
                                        f"**Q{len(all_results)}:** {item['query'][:100]}..."
                                        if len(item["query"]) > 100
                                        else f"**Q{len(all_results)}:** {item['query']}"
                                    )
            except json.JSONDecodeError:
                pass

        # Final progress update
        progress_bar.progress(1.0, text="Generation complete!")
        status_text.write(f"**Total questions generated:** {len(all_results)}")

        return all_results

    except Exception as e:
        st.error(f"Error generating chunk-level data: {e}")
        return None


def generate_token_level_data_with_progress(
    corpus: Corpus,
    queries_per_doc: int,
    api_key: str,
    progress_container,
    questions_container,
) -> list[TokenLevelGroundTruth] | None:
    """Generate token-level synthetic data with real-time progress updates.

    Args:
        corpus: The document corpus.
        queries_per_doc: Number of queries per document.
        api_key: OpenAI API key.
        progress_container: Streamlit container for progress bar.
        questions_container: Streamlit container for showing generated questions.

    Returns:
        List of TokenLevelGroundTruth objects or None if generation fails.
    """
    import hashlib
    import json
    from difflib import SequenceMatcher

    from openai import OpenAI

    from rag_evaluation_system.types import CharacterSpan, Query, QueryId, QueryText
    from rag_evaluation_system.types.ground_truth import TokenLevelGroundTruth

    def fuzzy_find(text: str, excerpt: str, threshold: float = 0.9) -> int:
        """Find excerpt in text using fuzzy matching."""
        excerpt_len = len(excerpt)
        if excerpt_len == 0 or excerpt_len > len(text):
            return -1

        best_ratio = 0.0
        best_pos = -1
        step = max(1, excerpt_len // 10)

        for i in range(0, len(text) - excerpt_len + 1, step):
            window = text[i : i + excerpt_len]
            ratio = SequenceMatcher(None, excerpt, window).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_pos = i

        if best_pos > 0 and best_ratio > threshold * 0.9:
            search_start = max(0, best_pos - step)
            search_end = min(len(text) - excerpt_len + 1, best_pos + step)
            for i in range(search_start, search_end):
                window = text[i : i + excerpt_len]
                ratio = SequenceMatcher(None, excerpt, window).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pos = i

        return best_pos if best_ratio >= threshold else -1

    try:
        client = OpenAI(api_key=api_key)

        all_results: list[TokenLevelGroundTruth] = []
        total_docs = len(corpus.documents)

        # Progress bar
        progress_bar = progress_container.progress(0, text="Starting generation...")
        status_text = progress_container.empty()

        for doc_idx, doc in enumerate(corpus.documents):
            # Check for stop request
            if check_stop_requested():
                progress_bar.progress(1.0, text="Generation stopped by user")
                status_text.write(f"**Stopped** - Generated {len(all_results)} questions")
                return all_results if all_results else None

            # Update progress
            progress = doc_idx / total_docs
            progress_bar.progress(progress, text=f"Processing document {doc_idx + 1}/{total_docs}")
            status_text.write(f"**Document:** {doc.id}")

            # Step 1: Generate questions
            query_prompt = f"""Document:
{doc.content[:8000]}

Generate exactly {queries_per_doc} diverse questions that can be answered using this document.

Return ONLY a valid JSON array of question strings."""

            query_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates diverse questions about document content. Return a JSON array of question strings.",
                    },
                    {"role": "user", "content": query_prompt},
                ],
                temperature=0.7,
            )

            query_text = query_response.choices[0].message.content or ""

            # Parse questions
            cleaned = query_text.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                start_idx = 1 if lines[0].startswith("```") else 0
                end_idx = len(lines)
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip() == "```":
                        end_idx = i
                        break
                cleaned = "\n".join(lines[start_idx:end_idx])

            try:
                questions = json.loads(cleaned)
                if not isinstance(questions, list):
                    continue
            except json.JSONDecodeError:
                continue

            # Step 2: For each question, extract excerpts
            for question in questions:
                # Check for stop request
                if check_stop_requested():
                    progress_bar.progress(1.0, text="Generation stopped by user")
                    status_text.write(f"**Stopped** - Generated {len(all_results)} questions")
                    return all_results if all_results else None

                if not isinstance(question, str):
                    continue

                excerpt_prompt = f"""Document:
{doc.content[:8000]}

Question: {question}

Extract the exact passages from the document that answer this question.
Copy the text VERBATIM - do not paraphrase.

Return ONLY a valid JSON array of excerpt strings."""

                excerpt_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that extracts relevant passages from documents. Copy text VERBATIM. Return a JSON array of excerpt strings.",
                        },
                        {"role": "user", "content": excerpt_prompt},
                    ],
                    temperature=0.7,
                )

                excerpt_text = excerpt_response.choices[0].message.content or ""

                # Parse excerpts
                cleaned = excerpt_text.strip()
                if cleaned.startswith("```"):
                    lines = cleaned.split("\n")
                    start_idx = 1 if lines[0].startswith("```") else 0
                    end_idx = len(lines)
                    for i in range(len(lines) - 1, -1, -1):
                        if lines[i].strip() == "```":
                            end_idx = i
                            break
                    cleaned = "\n".join(lines[start_idx:end_idx])

                try:
                    excerpts = json.loads(cleaned)
                    if not isinstance(excerpts, list):
                        continue
                except json.JSONDecodeError:
                    continue

                # Step 3: Find character positions
                spans: list[CharacterSpan] = []
                for excerpt in excerpts:
                    if not isinstance(excerpt, str):
                        continue

                    start = doc.content.find(excerpt)
                    if start == -1:
                        start = fuzzy_find(doc.content, excerpt)

                    if start != -1:
                        end = start + len(excerpt)
                        actual_text = doc.content[start:end]
                        spans.append(
                            CharacterSpan(
                                doc_id=doc.id,
                                start=start,
                                end=end,
                                text=actual_text,
                            )
                        )

                if spans:
                    query_id = QueryId(
                        f"query_{hashlib.sha256(question.encode()).hexdigest()[:12]}"
                    )
                    gt = TokenLevelGroundTruth(
                        query=Query(
                            id=query_id,
                            text=QueryText(question),
                            metadata={"source_doc": doc.id},
                        ),
                        relevant_spans=spans,
                    )
                    all_results.append(gt)

                    # Update questions display in real-time
                    with questions_container:
                        st.markdown(
                            f"**Q{len(all_results)}:** {question[:100]}..."
                            if len(question) > 100
                            else f"**Q{len(all_results)}:** {question}"
                        )

        # Final progress update
        progress_bar.progress(1.0, text="Generation complete!")
        status_text.write(f"**Total questions generated:** {len(all_results)}")

        return all_results

    except Exception as e:
        st.error(f"Error generating token-level data: {e}")
        return None


def generate_single_doc_chunk_level(
    corpus: Corpus,
    doc_idx: int,
    chunk_size: int,
    chunk_overlap: int,
    queries_per_doc: int,
    api_key: str,
) -> list[ChunkLevelGroundTruth]:
    """Generate chunk-level data for a single document.

    Args:
        corpus: The document corpus.
        doc_idx: Index of the document to process.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between chunks.
        queries_per_doc: Number of queries per document.
        api_key: OpenAI API key.

    Returns:
        List of ChunkLevelGroundTruth objects for this document.
    """
    import hashlib
    import json

    from openai import OpenAI

    from rag_evaluation_system.chunkers.recursive_character import RecursiveCharacterChunker
    from rag_evaluation_system.types import Query, QueryId, QueryText
    from rag_evaluation_system.types.ground_truth import ChunkLevelGroundTruth
    from rag_evaluation_system.utils.hashing import generate_chunk_id

    results: list[ChunkLevelGroundTruth] = []

    try:
        client = OpenAI(api_key=api_key)
        chunker = RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        doc = corpus.documents[doc_idx]

        # Build chunk index for this document
        chunk_index: dict[str, str] = {}
        doc_chunks = chunker.chunk(doc.content)
        for chunk_text in doc_chunks:
            cid = generate_chunk_id(chunk_text)
            chunk_index[cid] = chunk_text

        if not doc_chunks:
            return results

        # Format chunks for prompt
        chunks_with_ids = [(generate_chunk_id(ct), ct) for ct in doc_chunks]
        chunks_text = "\n\n".join(f"[{cid}]: {content}" for cid, content in chunks_with_ids[:20])

        system_prompt = """You are a helpful assistant that generates question-answer pairs for evaluating retrieval systems.

Given a set of text chunks with their IDs, generate questions that can be answered using the information in specific chunks.

Rules:
1. Generate diverse questions that cover different aspects of the content
2. Questions should be answerable using ONLY the provided chunks
3. Each question should cite the specific chunk IDs that contain the answer
4. Questions should be clear, specific, and unambiguous

Output Format:
Return a JSON array of objects, each with:
- "query": The question text
- "relevant_chunk_ids": Array of chunk IDs that contain the answer

Important: Only use chunk IDs that were provided to you. Do not invent chunk IDs."""

        user_prompt = f"""Here are the text chunks with their IDs:

{chunks_text}

Generate exactly {queries_per_doc} diverse questions that can be answered using these chunks.
Return ONLY a valid JSON array."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )

        response_text = response.choices[0].message.content or ""

        # Parse response
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            start_idx = 1 if lines[0].startswith("```") else 0
            end_idx = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == "```":
                    end_idx = i
                    break
            cleaned = "\n".join(lines[start_idx:end_idx])

        data = json.loads(cleaned)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "query" in item and "relevant_chunk_ids" in item:
                    valid_ids = [cid for cid in item["relevant_chunk_ids"] if cid in chunk_index]
                    if valid_ids:
                        query_id = QueryId(
                            f"query_{hashlib.sha256(item['query'].encode()).hexdigest()[:12]}"
                        )
                        gt = ChunkLevelGroundTruth(
                            query=Query(
                                id=query_id,
                                text=QueryText(item["query"]),
                                metadata={"source_doc": doc.id},
                            ),
                            relevant_chunk_ids=valid_ids,
                        )
                        results.append(gt)

    except Exception as e:
        st.warning(f"Error processing document {doc_idx}: {e}")

    return results


def generate_single_doc_token_level(
    corpus: Corpus,
    doc_idx: int,
    queries_per_doc: int,
    api_key: str,
) -> list[TokenLevelGroundTruth]:
    """Generate token-level data for a single document.

    Args:
        corpus: The document corpus.
        doc_idx: Index of the document to process.
        queries_per_doc: Number of queries per document.
        api_key: OpenAI API key.

    Returns:
        List of TokenLevelGroundTruth objects for this document.
    """
    import hashlib
    import json
    from difflib import SequenceMatcher

    from openai import OpenAI

    from rag_evaluation_system.types import CharacterSpan, Query, QueryId, QueryText
    from rag_evaluation_system.types.ground_truth import TokenLevelGroundTruth

    def fuzzy_find(text: str, excerpt: str, threshold: float = 0.9) -> int:
        excerpt_len = len(excerpt)
        if excerpt_len == 0 or excerpt_len > len(text):
            return -1

        best_ratio = 0.0
        best_pos = -1
        step = max(1, excerpt_len // 10)

        for i in range(0, len(text) - excerpt_len + 1, step):
            window = text[i : i + excerpt_len]
            ratio = SequenceMatcher(None, excerpt, window).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_pos = i

        if best_pos > 0 and best_ratio > threshold * 0.9:
            search_start = max(0, best_pos - step)
            search_end = min(len(text) - excerpt_len + 1, best_pos + step)
            for i in range(search_start, search_end):
                window = text[i : i + excerpt_len]
                ratio = SequenceMatcher(None, excerpt, window).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pos = i

        return best_pos if best_ratio >= threshold else -1

    results: list[TokenLevelGroundTruth] = []

    try:
        client = OpenAI(api_key=api_key)
        doc = corpus.documents[doc_idx]

        # Step 1: Generate questions
        query_prompt = f"""Document:
{doc.content[:8000]}

Generate exactly {queries_per_doc} diverse questions that can be answered using this document.

Return ONLY a valid JSON array of question strings."""

        query_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates diverse questions about document content. Return a JSON array of question strings.",
                },
                {"role": "user", "content": query_prompt},
            ],
            temperature=0.7,
        )

        query_text = query_response.choices[0].message.content or ""

        # Parse questions
        cleaned = query_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            start_idx = 1 if lines[0].startswith("```") else 0
            end_idx = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == "```":
                    end_idx = i
                    break
            cleaned = "\n".join(lines[start_idx:end_idx])

        questions = json.loads(cleaned)
        if not isinstance(questions, list):
            return results

        # Step 2: For each question, extract excerpts
        for question in questions:
            if not isinstance(question, str):
                continue

            excerpt_prompt = f"""Document:
{doc.content[:8000]}

Question: {question}

Extract the exact passages from the document that answer this question.
Copy the text VERBATIM - do not paraphrase.

Return ONLY a valid JSON array of excerpt strings."""

            excerpt_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that extracts relevant passages from documents. Copy text VERBATIM. Return a JSON array of excerpt strings.",
                    },
                    {"role": "user", "content": excerpt_prompt},
                ],
                temperature=0.7,
            )

            excerpt_text = excerpt_response.choices[0].message.content or ""

            # Parse excerpts
            cleaned = excerpt_text.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                start_idx = 1 if lines[0].startswith("```") else 0
                end_idx = len(lines)
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip() == "```":
                        end_idx = i
                        break
                cleaned = "\n".join(lines[start_idx:end_idx])

            try:
                excerpts = json.loads(cleaned)
                if not isinstance(excerpts, list):
                    continue
            except json.JSONDecodeError:
                continue

            # Step 3: Find character positions
            spans: list[CharacterSpan] = []
            for excerpt in excerpts:
                if not isinstance(excerpt, str):
                    continue

                start = doc.content.find(excerpt)
                if start == -1:
                    start = fuzzy_find(doc.content, excerpt)

                if start != -1:
                    end = start + len(excerpt)
                    actual_text = doc.content[start:end]
                    spans.append(
                        CharacterSpan(
                            doc_id=doc.id,
                            start=start,
                            end=end,
                            text=actual_text,
                        )
                    )

            if spans:
                query_id = QueryId(
                    f"query_{hashlib.sha256(question.encode()).hexdigest()[:12]}"
                )
                gt = TokenLevelGroundTruth(
                    query=Query(
                        id=query_id,
                        text=QueryText(question),
                        metadata={"source_doc": doc.id},
                    ),
                    relevant_spans=spans,
                )
                results.append(gt)

    except Exception as e:
        st.warning(f"Error processing document {doc_idx}: {e}")

    return results


def render_document_sidebar(corpus: Corpus) -> None:
    """Render document list in sidebar.

    Args:
        corpus: The loaded corpus.
    """
    st.sidebar.header("Documents")
    st.sidebar.write(f"**{len(corpus.documents)}** documents loaded")

    for doc in corpus.documents:
        with st.sidebar.expander(f"{doc.id} ({doc.char_count:,} chars)"):
            # Preview options
            view_mode = st.radio(
                "View mode",
                ["Markdown", "Raw"],
                key=f"view_mode_{doc.id}",
                horizontal=True,
            )

            if view_mode == "Markdown":
                st.markdown(doc.content[:2000] + ("..." if len(doc.content) > 2000 else ""))
            else:
                st.code(doc.content[:2000] + ("..." if len(doc.content) > 2000 else ""))


def render_chunk_level_results(
    ground_truth: list[ChunkLevelGroundTruth],
    corpus: Corpus,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Render chunk-level generation results.

    Args:
        ground_truth: List of generated ground truth objects.
        corpus: The document corpus.
        chunk_size: Chunk size used for generation.
        chunk_overlap: Chunk overlap used for generation.
    """
    from rag_evaluation_system.chunkers.recursive_character import RecursiveCharacterChunker
    from rag_evaluation_system.utils.hashing import generate_chunk_id

    st.success(f"Generated {len(ground_truth)} questions!")

    # Build chunk index for lookup
    chunker = RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunk_index: dict[str, str] = {}
    for doc in corpus.documents:
        chunk_texts = chunker.chunk(doc.content)
        for chunk_text in chunk_texts:
            chunk_id = generate_chunk_id(chunk_text)
            chunk_index[chunk_id] = chunk_text

    # Question selector
    question_options = [f"Q{i + 1}: {gt.query.text[:60]}..." for i, gt in enumerate(ground_truth)]

    selected_idx = st.selectbox(
        "Select a question to view details",
        range(len(ground_truth)),
        format_func=lambda i: question_options[i],
    )

    if selected_idx is not None:
        selected_gt = ground_truth[selected_idx]

        st.subheader("Question")
        st.info(selected_gt.query.text)

        st.subheader(f"Relevant Chunks ({len(selected_gt.relevant_chunk_ids)})")

        for chunk_id in selected_gt.relevant_chunk_ids:
            chunk_content = chunk_index.get(chunk_id, "Chunk not found")
            with st.expander(f"Chunk: {chunk_id}"):
                st.code(chunk_content, language=None)


def render_token_level_results(
    ground_truth: list[TokenLevelGroundTruth],
    corpus: Corpus,
) -> None:
    """Render token-level generation results.

    Args:
        ground_truth: List of generated ground truth objects.
        corpus: The document corpus.
    """
    st.success(f"Generated {len(ground_truth)} questions!")

    # Question selector
    question_options = [f"Q{i + 1}: {gt.query.text[:60]}..." for i, gt in enumerate(ground_truth)]

    selected_idx = st.selectbox(
        "Select a question to view details",
        range(len(ground_truth)),
        format_func=lambda i: question_options[i],
    )

    if selected_idx is not None:
        selected_gt = ground_truth[selected_idx]

        st.subheader("Question")
        st.info(selected_gt.query.text)

        st.subheader(f"Relevant Spans ({len(selected_gt.relevant_spans)})")

        # Group spans by document
        spans_by_doc: dict[str, list[CharacterSpan]] = {}
        for span in selected_gt.relevant_spans:
            if span.doc_id not in spans_by_doc:
                spans_by_doc[span.doc_id] = []
            spans_by_doc[span.doc_id].append(span)

        # Render each document with highlighted spans
        for doc_id, spans in spans_by_doc.items():
            doc = corpus.get_document(doc_id)
            if doc is None:
                st.warning(f"Document not found: {doc_id}")
                continue

            with st.expander(f"Document: {doc_id} ({len(spans)} span(s))", expanded=True):
                # Show span details
                st.write("**Highlighted Spans:**")
                for i, span in enumerate(spans):
                    st.write(f"- Span {i + 1}: chars {span.start}-{span.end}")

                # Show document with highlighted spans
                st.write("**Document with Highlights:**")
                highlighted_html = highlight_spans_in_document(doc.content, spans)
                st.markdown(highlighted_html, unsafe_allow_html=True)


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="RAG Evaluation System",
        page_icon="magnifier",
        layout="wide",
    )

    st.title("RAG Evaluation System")
    st.write("Generate synthetic evaluation data for RAG systems.")

    # Initialize session state
    if "corpus" not in st.session_state:
        st.session_state.corpus = None
    if "chunk_level_results" not in st.session_state:
        st.session_state.chunk_level_results = None
    if "token_level_results" not in st.session_state:
        st.session_state.token_level_results = None

    # Mode selection
    st.header("1. Select Evaluation Mode")
    mode = st.radio(
        "Evaluation Type",
        ["Token-Level Evaluation (Recommended)", "Chunk-Level Evaluation"],
        help=(
            "Token-level evaluation is chunker-independent and recommended for fair comparison "
            "across different chunking strategies. Chunk-level evaluation is simpler but ties "
            "evaluation to a specific chunking strategy."
        ),
    )
    is_token_level = mode == "Token-Level Evaluation (Recommended)"

    # Chunker configuration (only for chunk-level)
    if not is_token_level:
        st.header("2. Configure Chunker")
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input(
                "Chunk Size (characters)",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                help="Target size for each chunk in characters.",
            )
        with col2:
            chunk_overlap = st.number_input(
                "Chunk Overlap (characters)",
                min_value=0,
                max_value=int(chunk_size) - 1,
                value=min(200, int(chunk_size) - 1),
                step=50,
                help="Number of characters to overlap between chunks.",
            )
    else:
        chunk_size = 1000
        chunk_overlap = 200

    # Corpus loading
    section_num = "2" if is_token_level else "3"
    st.header(f"{section_num}. Load Corpus")

    # Get suggested folders
    suggested_folders = get_suggested_folders()

    # Show folder suggestions if available
    if suggested_folders:
        st.write("**Quick Select** - Click to select a corpus:")
        cols = st.columns(min(len(suggested_folders), 4))
        for i, (folder_path, label) in enumerate(suggested_folders[:4]):
            with cols[i]:
                if st.button(f"📁 {label}", key=f"quick_select_{i}", use_container_width=True):
                    st.session_state.folder_path = folder_path
                    st.session_state.corpus = None
                    st.session_state.chunk_level_results = None
                    st.session_state.token_level_results = None
                    st.rerun()

        st.write("**Or** enter a custom path:")

    # Text input for folder path
    folder_path = st.text_input(
        "Folder Path",
        value=st.session_state.get("folder_path", ""),
        placeholder="/path/to/your/markdown/files",
        help="Enter the path to a folder containing markdown (.md) files.",
    )

    # Update session state if text input changed
    if folder_path != st.session_state.get("folder_path", ""):
        st.session_state.folder_path = folder_path

    # Show Load Corpus button
    if folder_path:
        load_button = st.button("Load Corpus", type="primary")

        if load_button:
            with st.spinner("Loading corpus..."):
                corpus = load_corpus(folder_path)
                if corpus:
                    st.session_state.corpus = corpus
                    # Clear previous results when loading new corpus
                    st.session_state.chunk_level_results = None
                    st.session_state.token_level_results = None
                    st.success(f"Loaded {len(corpus.documents)} documents!")

    # Show document sidebar if corpus is loaded
    if st.session_state.corpus is not None:
        render_document_sidebar(st.session_state.corpus)

        # Generation settings
        section_num = "3" if is_token_level else "4"
        st.header(f"{section_num}. Generation Settings")

        queries_per_doc = st.number_input(
            "Questions per Document",
            min_value=1,
            max_value=50,
            value=10,
            help="Number of synthetic questions to generate per document.",
        )

        # Get API key from environment
        api_key = os.environ.get("OPENAI_API_KEY", "")

        # Generate button
        section_num = "4" if is_token_level else "5"
        st.header(f"{section_num}. Generate Questions")

        if not api_key:
            st.warning(
                "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file. "
                "See .env.example for reference."
            )
        else:
            # Initialize generation state
            if "generation_in_progress" not in st.session_state:
                st.session_state.generation_in_progress = False
            if "stop_generation" not in st.session_state:
                st.session_state.stop_generation = False
            if "pending_results" not in st.session_state:
                st.session_state.pending_results = []
            if "current_doc_idx" not in st.session_state:
                st.session_state.current_doc_idx = 0

            # Create button columns
            col1, col2 = st.columns([1, 1])

            with col1:
                generate_button = st.button(
                    "Generate Questions",
                    type="primary",
                    disabled=st.session_state.generation_in_progress,
                )

            with col2:
                stop_button = st.button(
                    "Stop Generation",
                    type="secondary",
                    on_click=set_stop_flag,
                )

            # Handle stop - finalize results and reset state
            if st.session_state.stop_generation and st.session_state.generation_in_progress:
                st.warning("Generation stopped by user.")
                if st.session_state.pending_results:
                    if is_token_level:
                        st.session_state.token_level_results = st.session_state.pending_results
                    else:
                        st.session_state.chunk_level_results = st.session_state.pending_results
                        st.session_state.chunk_size = int(chunk_size)
                        st.session_state.chunk_overlap = int(chunk_overlap)
                    st.success(f"Saved {len(st.session_state.pending_results)} questions generated so far.")
                # Reset state
                st.session_state.generation_in_progress = False
                st.session_state.stop_generation = False
                st.session_state.pending_results = []
                st.session_state.current_doc_idx = 0

            # Start new generation
            if generate_button:
                # Reset state for new generation
                st.session_state.stop_generation = False
                st.session_state.generation_in_progress = True
                st.session_state.pending_results = []
                st.session_state.current_doc_idx = 0
                if is_token_level:
                    st.session_state.token_level_results = None
                else:
                    st.session_state.chunk_level_results = None
                st.rerun()

            # Continue generation if in progress
            if st.session_state.generation_in_progress and not st.session_state.stop_generation:
                corpus = st.session_state.corpus

                # Create containers for real-time progress display
                progress_container = st.container()
                questions_container = st.container()

                # Show progress
                total_docs = len(corpus.documents)
                doc_idx = st.session_state.current_doc_idx

                progress_container.progress(
                    doc_idx / total_docs,
                    text=f"Processing document {doc_idx + 1}/{total_docs}..."
                )

                # Show previously generated questions
                with questions_container:
                    st.subheader(f"Generated Questions ({len(st.session_state.pending_results)} so far)")
                    for i, gt in enumerate(st.session_state.pending_results[-5:]):  # Show last 5
                        st.markdown(f"**Q{len(st.session_state.pending_results) - 4 + i}:** {gt.query.text[:100]}...")

                if doc_idx < total_docs:
                    # Generate for current document
                    if is_token_level:
                        new_results = generate_single_doc_token_level(
                            corpus=corpus,
                            doc_idx=doc_idx,
                            queries_per_doc=int(queries_per_doc),
                            api_key=api_key,
                        )
                    else:
                        new_results = generate_single_doc_chunk_level(
                            corpus=corpus,
                            doc_idx=doc_idx,
                            chunk_size=int(chunk_size),
                            chunk_overlap=int(chunk_overlap),
                            queries_per_doc=int(queries_per_doc),
                            api_key=api_key,
                        )

                    if new_results:
                        st.session_state.pending_results.extend(new_results)

                    # Move to next document
                    st.session_state.current_doc_idx += 1
                    st.rerun()
                else:
                    # Generation complete
                    if st.session_state.pending_results:
                        if is_token_level:
                            st.session_state.token_level_results = st.session_state.pending_results
                        else:
                            st.session_state.chunk_level_results = st.session_state.pending_results
                            st.session_state.chunk_size = int(chunk_size)
                            st.session_state.chunk_overlap = int(chunk_overlap)

                    st.session_state.generation_in_progress = False
                    st.session_state.pending_results = []
                    st.session_state.current_doc_idx = 0
                    st.rerun()

        # Display results
        section_num = "5" if is_token_level else "6"
        st.header(f"{section_num}. Results")

        if is_token_level and st.session_state.token_level_results:
            render_token_level_results(
                st.session_state.token_level_results,
                st.session_state.corpus,
            )
        elif not is_token_level and st.session_state.chunk_level_results:
            render_chunk_level_results(
                st.session_state.chunk_level_results,
                st.session_state.corpus,
                st.session_state.chunk_size,
                st.session_state.chunk_overlap,
            )
        else:
            st.info("No results yet. Load a corpus and generate questions to see results here.")


if __name__ == "__main__":
    main()
