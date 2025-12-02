from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple

from src.normalization.dictionaries import DictionaryLoader


def _expand_labels_for_id(entry_id: str, dl: DictionaryLoader) -> Set[str]:
    entry = dl.get_entry(entry_id)
    if entry:
        return set(x.lower() for x in entry.all_labels())
    return set()


def raf_score(pred_ids: Sequence[str], evidence_texts: Sequence[str], dl: DictionaryLoader | None = None,
              token_overlap_threshold: float = 0.0, use_label_ngram: bool = False) -> float:
    """Reference-Aware Factuality: fraction of predicted IDs supported by evidence.

    Supported if any of the ID's labels/synonyms appears in any evidence text (case-insensitive substring).
    """
    if not pred_ids:
        return 0.0
    dl = dl or DictionaryLoader()
    if not pred_ids:
        return 0.0
    dl = dl or DictionaryLoader()
    supported = 0
    ev_lowers = [e.lower() for e in evidence_texts]
    for pid in pred_ids:
        labels = _expand_labels_for_id(pid, dl)
        if not labels:
            continue
        # 1. Exact substring match
        ok = any(any(lbl in ev for lbl in labels) for ev in ev_lowers)
        
        if not ok and token_overlap_threshold > 0.0:
            # fallback: token overlap (Jaccard) between label tokens and evidence tokens
            import re
            def toks(s: str) -> Set[str]:
                return set(re.findall(r"[a-z0-9]+", s))
            label_tokens = set().union(*[toks(l) for l in labels])
            for ev in ev_lowers:
                ev_tokens = toks(ev)
                inter = len(label_tokens & ev_tokens)
                uni = len(label_tokens | ev_tokens) or 1
                j = inter / uni
                if j >= token_overlap_threshold:
                    ok = True
                    break
        
        if not ok and use_label_ngram:
            # n-gram(1..3-gram) match
            import re
            def ngrams(words: List[str], n: int) -> List[str]:
                return [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
            label_ngrams: Set[str] = set()
            for l in labels:
                ws = re.findall(r"[a-z0-9]+", l)
                for n in (3, 2, 1):
                    for g in ngrams(ws, n):
                        if len(g) >= 3:
                            label_ngrams.add(g)
            for ev in ev_lowers:
                for g in label_ngrams:
                    if g in ev:
                        ok = True
                        break
                if ok:
                    break
                    
        if ok:
            supported += 1
    return supported / max(1, len(pred_ids))


def pkg_score(final_ids: Sequence[str], evidence_texts: Sequence[str], dl: DictionaryLoader | None = None,
              token_overlap_threshold: float = 0.0, use_label_ngram: bool = False) -> float:
    """Principal Knowledge Grounding: proportion of final answer IDs that are grounded in evidence.

    Here we reuse the same matching rule as RAF, applied to final IDs.
    """
    return raf_score(final_ids, evidence_texts, dl, token_overlap_threshold, use_label_ngram)


