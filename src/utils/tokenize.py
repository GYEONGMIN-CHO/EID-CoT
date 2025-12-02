from __future__ import annotations

import re
from typing import List, Sequence

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lightweight alphanumeric tokenizer suitable for BM25 examples."""
    return _WORD_RE.findall(text.lower())


def tokenize_corpus(corpus: Sequence[str]) -> List[List[str]]:
    return [tokenize(doc) for doc in corpus]


