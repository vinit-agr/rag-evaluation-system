import json
import math
from typing import Any, Iterable

import streamlit as st

from rag_evaluation_system.chunkers.recursive_character import RecursiveCharacterChunker
from rag_evaluation_system.evaluation.metrics.chunk_level import (
    ChunkF1,
    ChunkPrecision,
    ChunkRecall,
)
from rag_evaluation_system.evaluation.metrics.token_level import (
    SpanIoU,
    SpanPrecision,
    SpanRecall,
)
from rag_evaluation_system.types import (
    CharacterSpan,
    Document,
    DocumentId,
    PositionAwareChunk,
)
from rag_evaluation_system.utils.hashing import generate_chunk_id


st.set_page_config(page_title="RAG Eval Smoke Test", layout="wide")

st.title("RAG Evaluation System - Smoke Test")
st.caption("Lightweight UI for testing chunk and span metrics locally.")

st.sidebar.header("Document")

raw_doc = st.sidebar.text_area(
    "Document content",
    value="Paste a document here to generate chunks and spans.",
    height=200,
)

chunk_size = st.sidebar.number_input("Chunk size", min_value=50, max_value=4000, value=400)
chunk_overlap = st.sidebar.number_input(
    "Chunk overlap", min_value=0, max_value=int(chunk_size) - 1, value=50
)

if chunk_overlap >= chunk_size:
    st.sidebar.error("Chunk overlap must be less than chunk size.")


class SimpleEmbedder:
    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension

    @property
    def name(self) -> str:
        return f"SimpleEmbedder(dim={self._dimension})"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._hash_embed(query)

    def _hash_embed(self, text: str) -> list[float]:
        buckets = [0.0] * self._dimension
        for token in text.lower().split():
            idx = hash(token) % self._dimension
            buckets[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in buckets)) or 1.0
        return [v / norm for v in buckets]


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._entries: list[tuple[PositionAwareChunk, list[float]]] = []

    @property
    def name(self) -> str:
        return "InMemoryVectorStore"

    def add(self, chunks: list[PositionAwareChunk], embeddings: list[list[float]]) -> None:
        self._entries.extend(list(zip(chunks, embeddings)))

    def search(self, query_embedding: list[float], k: int = 5) -> list[PositionAwareChunk]:
        scored = [
            (cosine_similarity(query_embedding, embedding), chunk)
            for chunk, embedding in self._entries
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:k]]

    def clear(self) -> None:
        self._entries = []


@st.cache_data(show_spinner=False)
def build_chunks(text: str, size: int, overlap: int) -> list[dict[str, Any]]:
    chunker = RecursiveCharacterChunker(chunk_size=size, chunk_overlap=overlap)
    doc = Document(id=DocumentId("doc_1"), content=text)
    chunks = chunker.chunk(doc.content)
    return [
        {"id": generate_chunk_id(chunk), "content": chunk}
        for chunk in chunks
    ]


@st.cache_data(show_spinner=False)
def build_position_aware(text: str, size: int, overlap: int):
    chunker = RecursiveCharacterChunker(chunk_size=size, chunk_overlap=overlap)
    doc = Document(id=DocumentId("doc_1"), content=text)
    return chunker.chunk_with_positions(doc)


def chunk_preview(text: str, limit: int = 120) -> str:
    preview = text.replace("\n", " ")
    return preview if len(preview) <= limit else f"{preview[:limit]}..."


doc = Document(id=DocumentId("doc_1"), content=raw_doc)

chunk_tab, span_tab = st.tabs(["Chunk-Level Metrics", "Token-Level Metrics"])

with chunk_tab:
    st.subheader("Chunk-Level Evaluation")
    st.caption("Select ground truth and retrieved chunks manually or run a simple retrieval.")

    if not raw_doc.strip():
        st.info("Enter document content to generate chunks.")
    else:
        chunks = build_chunks(raw_doc, int(chunk_size), int(chunk_overlap))
        if not chunks:
            st.warning("No chunks generated.")
        else:
            options = [
                f"{c['id']} | {chunk_preview(c['content'])}" for c in chunks
            ]
            id_map = {options[i]: chunks[i]["id"] for i in range(len(chunks))}

            col1, col2 = st.columns(2)
            with col1:
                gt_selection = st.multiselect(
                    "Ground truth chunk IDs",
                    options,
                )
            with col2:
                retrieved_selection = st.multiselect(
                    "Retrieved chunk IDs",
                    options,
                )

            st.markdown("**Quick retrieval**")
            query_text = st.text_input("Query", value="What is this document about?")
            k_retrieve = st.number_input("Top K", min_value=1, max_value=10, value=3)

            if st.button("Retrieve chunks"):
                pa_chunks = build_position_aware(raw_doc, int(chunk_size), int(chunk_overlap))
                embedder = SimpleEmbedder()
                store = InMemoryVectorStore()
                embeddings = embedder.embed([chunk.content for chunk in pa_chunks])
                store.add(pa_chunks, embeddings)
                retrieved = store.search(embedder.embed_query(query_text), int(k_retrieve))
                retrieved_selection = [
                    f"{generate_chunk_id(chunk.content)} | {chunk_preview(chunk.content)}"
                    for chunk in retrieved
                ]
                st.session_state["retrieved_selection"] = retrieved_selection

            if "retrieved_selection" in st.session_state:
                retrieved_selection = st.session_state["retrieved_selection"]

            if st.button("Compute chunk metrics"):
                gt_ids = [id_map[item] for item in gt_selection]
                retrieved_ids = [id_map[item] for item in retrieved_selection]

                recall = ChunkRecall().calculate(retrieved_ids, gt_ids)
                precision = ChunkPrecision().calculate(retrieved_ids, gt_ids)
                f1 = ChunkF1().calculate(retrieved_ids, gt_ids)

                st.success("Metrics computed")
                st.metric("Chunk Recall", f"{recall:.3f}")
                st.metric("Chunk Precision", f"{precision:.3f}")
                st.metric("Chunk F1", f"{f1:.3f}")

            with st.expander("Show chunks"):
                for chunk in chunks:
                    st.write(f"{chunk['id']}: {chunk_preview(chunk['content'], 200)}")

with span_tab:
    st.subheader("Token-Level (Character Span) Evaluation")
    st.caption("Uses position-aware chunks from the same document.")

    if not raw_doc.strip():
        st.info("Enter document content to generate position-aware chunks.")
    else:
        pa_chunks = build_position_aware(raw_doc, int(chunk_size), int(chunk_overlap))
        options = [
            f"{c.id} | {c.start}-{c.end} | {chunk_preview(c.content)}" for c in pa_chunks
        ]
        id_map = {options[i]: pa_chunks[i] for i in range(len(pa_chunks))}

        retrieved_selection = st.multiselect("Retrieved chunks", options)
        retrieved_spans = [id_map[item].to_span() for item in retrieved_selection]

        st.markdown("**Ground truth spans (JSON)**")
        st.caption(
            "Provide a JSON list like: [{\"start\": 0, \"end\": 42}, ...]. "
            "Text is filled from the document automatically."
        )
        spans_json = st.text_area(
            "Ground truth spans",
            value="[]",
            height=150,
        )

        st.markdown("**Quick retrieval**")
        query_text = st.text_input("Query", value="What is this document about?", key="span_query")
        k_retrieve = st.number_input("Top K", min_value=1, max_value=10, value=3, key="span_k")

        if st.button("Retrieve spans", key="retrieve_spans"):
            embedder = SimpleEmbedder()
            store = InMemoryVectorStore()
            embeddings = embedder.embed([chunk.content for chunk in pa_chunks])
            store.add(pa_chunks, embeddings)
            retrieved = store.search(embedder.embed_query(query_text), int(k_retrieve))
            retrieved_selection = [
                f"{chunk.id} | {chunk.start}-{chunk.end} | {chunk_preview(chunk.content)}"
                for chunk in retrieved
            ]
            st.session_state["retrieved_span_selection"] = retrieved_selection

        if "retrieved_span_selection" in st.session_state:
            retrieved_selection = st.session_state["retrieved_span_selection"]
            retrieved_spans = [id_map[item].to_span() for item in retrieved_selection]

        if st.button("Compute span metrics"):
            try:
                data = json.loads(spans_json)
                if not isinstance(data, list):
                    raise ValueError("Ground truth must be a JSON list.")

                gt_spans: list[CharacterSpan] = []
                for item in data:
                    start = int(item["start"])
                    end = int(item["end"])
                    if start < 0 or end > len(doc.content):
                        raise ValueError("Span boundaries are out of range.")
                    gt_spans.append(
                        CharacterSpan(
                            doc_id=doc.id,
                            start=start,
                            end=end,
                            text=doc.content[start:end],
                        )
                    )

                recall = SpanRecall().calculate(retrieved_spans, gt_spans)
                precision = SpanPrecision().calculate(retrieved_spans, gt_spans)
                iou = SpanIoU().calculate(retrieved_spans, gt_spans)

                st.success("Metrics computed")
                st.metric("Span Recall", f"{recall:.3f}")
                st.metric("Span Precision", f"{precision:.3f}")
                st.metric("Span IoU", f"{iou:.3f}")
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                st.error(f"Invalid ground truth spans: {exc}")

        with st.expander("Show retrieved spans"):
            for span in retrieved_spans:
                st.write(f"{span.start}-{span.end}: {chunk_preview(span.text, 200)}")
