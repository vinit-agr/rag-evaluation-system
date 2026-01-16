"""Utility functions for token-level (character span) metrics.

This module provides functions for merging overlapping spans and calculating
character-level overlap between span collections. These utilities ensure that
each character is counted at most once, preventing sliding-window chunkers
from inflating metrics.
"""

from collections import defaultdict

from rag_evaluation_system.types import CharacterSpan, DocumentId


def merge_overlapping_spans(spans: list[CharacterSpan]) -> list[CharacterSpan]:
    """Merge overlapping or adjacent spans within each document.

    Groups spans by document ID, sorts by start position, and merges any
    overlapping or adjacent intervals. The merged spans have placeholder
    text since we cannot reconstruct the actual text without the source.

    Args:
        spans: List of character spans to merge.

    Returns:
        List of merged spans with no overlaps. Each merged span has
        placeholder text ("_" repeated for the span length).

    Example:
        Input spans (same doc):
            - (0, 10, "aaaaaaaaaa")
            - (5, 15, "bbbbbbbbbb")
            - (20, 30, "cccccccccc")
        Output:
            - (0, 15, "_______________")  # merged first two
            - (20, 30, "__________")      # unchanged
    """
    if not spans:
        return []

    # Group spans by document ID
    spans_by_doc: dict[DocumentId, list[CharacterSpan]] = defaultdict(list)
    for span in spans:
        spans_by_doc[span.doc_id].append(span)

    merged_spans: list[CharacterSpan] = []

    for doc_id, doc_spans in spans_by_doc.items():
        # Sort by start position
        sorted_spans = sorted(doc_spans, key=lambda s: s.start)

        # Merge overlapping intervals
        current_start = sorted_spans[0].start
        current_end = sorted_spans[0].end

        for span in sorted_spans[1:]:
            if span.start <= current_end:
                # Overlapping or adjacent - extend the current interval
                current_end = max(current_end, span.end)
            else:
                # No overlap - save the current interval and start a new one
                merged_length = current_end - current_start
                merged_spans.append(
                    CharacterSpan(
                        doc_id=doc_id,
                        start=current_start,
                        end=current_end,
                        text="_" * merged_length,
                    )
                )
                current_start = span.start
                current_end = span.end

        # Don't forget the last interval
        merged_length = current_end - current_start
        merged_spans.append(
            CharacterSpan(
                doc_id=doc_id,
                start=current_start,
                end=current_end,
                text="_" * merged_length,
            )
        )

    return merged_spans


def calculate_overlap(
    spans_a: list[CharacterSpan],
    spans_b: list[CharacterSpan],
) -> int:
    """Calculate the total character overlap between two span collections.

    Merges each collection first to avoid double-counting, then sums the
    overlap between all pairs of spans with the same document ID.

    Args:
        spans_a: First collection of character spans.
        spans_b: Second collection of character spans.

    Returns:
        Total number of overlapping characters across all documents.

    Example:
        spans_a: [(doc1, 0, 10), (doc1, 5, 15)]  -> merged: [(doc1, 0, 15)]
        spans_b: [(doc1, 10, 20)]                -> merged: [(doc1, 10, 20)]
        overlap: characters 10-15 = 5 characters
    """
    if not spans_a or not spans_b:
        return 0

    # Merge each collection
    merged_a = merge_overlapping_spans(spans_a)
    merged_b = merge_overlapping_spans(spans_b)

    # Group merged spans by document
    a_by_doc: dict[DocumentId, list[CharacterSpan]] = defaultdict(list)
    for span in merged_a:
        a_by_doc[span.doc_id].append(span)

    b_by_doc: dict[DocumentId, list[CharacterSpan]] = defaultdict(list)
    for span in merged_b:
        b_by_doc[span.doc_id].append(span)

    # Calculate total overlap
    total_overlap = 0
    for doc_id in a_by_doc:
        if doc_id not in b_by_doc:
            continue

        for span_a in a_by_doc[doc_id]:
            for span_b in b_by_doc[doc_id]:
                total_overlap += span_a.overlap_chars(span_b)

    return total_overlap


__all__ = [
    "calculate_overlap",
    "merge_overlapping_spans",
]
