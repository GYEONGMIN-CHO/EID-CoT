from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Evidence:
    doc_id: str
    span: str


@dataclass
class Step:
    idx: int
    ids: List[str]
    evidence: List[Evidence] = field(default_factory=list)
    claim: Optional[str] = None


@dataclass
class StepLog:
    qid: str
    steps: List[Step]
    final_answer: str
    citations: List[str] = field(default_factory=list)


