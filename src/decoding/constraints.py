from __future__ import annotations

import re
from typing import Iterable, List

ID_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"^mesh:D\d{6}$"),
    re.compile(r"^HGNC:\d+$"),
    re.compile(r"^GeneID:\d+$"),
    # Optional: DrugBank like IDs (license caution)
    re.compile(r"^DB\d{5}$"),
]


def is_valid_id(token: str) -> bool:
    """Return True if token matches one of the allowed ID patterns."""
    for pattern in ID_PATTERNS:
        if pattern.match(token):
            return True
    return False


def filter_invalid(tokens: Iterable[str]) -> List[str]:
    """Filter tokens, keeping only valid ID-like tokens.

    Note: This is a lightweight utility; decoding-time constraints should be enforced
    via DFA/regex-constrained decoding in the generation loop.
    """
    return [t for t in tokens if is_valid_id(t)]


