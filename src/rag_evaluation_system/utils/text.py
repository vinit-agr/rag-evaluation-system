"""Text processing utilities."""
import re


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace into single spaces and trim."""
    return re.sub(r"\s+", " ", text).strip()
