from __future__ import annotations

from typing import Iterable, Set, Tuple


def attribution_precision_recall(pred_citations: Iterable[str], gold_citations: Iterable[str]) -> Tuple[float, float]:
    pred: Set[str] = set(pred_citations)
    gold: Set[str] = set(gold_citations)
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    return precision, recall


