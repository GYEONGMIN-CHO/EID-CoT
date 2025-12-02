from __future__ import annotations

from typing import Iterable, Tuple


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 with safe zero-division handling."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def entity_grounding_f1(pred_ids: Iterable[str], gold_ids: Iterable[str]) -> Tuple[float, float, float]:
    """Entity Grounding F1 using exact normalized ID matching."""
    pred_set = set(pred_ids)
    gold_set = set(gold_ids)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return precision_recall_f1(tp, fp, fn)


