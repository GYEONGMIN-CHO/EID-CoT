from __future__ import annotations

from typing import List, Protocol

from .entity import Entity
from .scispacy_biosyn_provider import ScispaCyBioSynProvider


class NormalizationProvider(Protocol):
    def annotate(self, text: str) -> List[Entity]:
        ...


def build_provider(kind: str = "scispacy_biosyn") -> NormalizationProvider:
    kind = (kind or "scispacy_biosyn").lower()
    if kind == "scispacy_biosyn":
        return ScispaCyBioSynProvider()
    raise ValueError(f"Unknown normalization provider: {kind}")


