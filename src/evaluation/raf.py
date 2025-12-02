from __future__ import annotations

from typing import Iterable, Tuple


def relevance_aware_factuality(num_supported: int, num_claims: int) -> float:
    """RAF-style placeholder: fraction of claims that are supported by relevant passages.

    This is a simplified stand-in until a full judge/relevance module is implemented.
    """
    if num_claims <= 0:
        return 0.0
    return max(0.0, min(1.0, num_supported / num_claims))


