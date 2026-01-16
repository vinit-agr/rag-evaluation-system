"""Utilities for span operations."""
from rag_evaluation_system.types import CharacterSpan, DocumentId


def merge_overlapping_spans(spans: list[CharacterSpan]) -> list[CharacterSpan]:
    """Merge overlapping spans within each document."""
    if not spans:
        return []

    by_doc: dict[DocumentId, list[CharacterSpan]] = {}
    for span in spans:
        by_doc.setdefault(span.doc_id, []).append(span)

    merged: list[CharacterSpan] = []

    for doc_spans in by_doc.values():
        sorted_spans = sorted(doc_spans, key=lambda s: s.start)
        current = sorted_spans[0]

        for span in sorted_spans[1:]:
            if span.start <= current.end:
                current = CharacterSpan(
                    doc_id=current.doc_id,
                    start=current.start,
                    end=max(current.end, span.end),
                    text="",
                )
            else:
                merged.append(current)
                current = span

        merged.append(current)

    return merged


def calculate_overlap(spans_a: list[CharacterSpan], spans_b: list[CharacterSpan]) -> int:
    """Calculate total character overlap between two span lists."""
    merged_a = merge_overlapping_spans(spans_a)
    merged_b = merge_overlapping_spans(spans_b)

    total_overlap = 0
    for span_a in merged_a:
        for span_b in merged_b:
            if span_a.doc_id == span_b.doc_id:
                total_overlap += span_a.overlap_chars(span_b)

    return total_overlap
