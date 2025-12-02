from __future__ import annotations

import csv
import os
from typing import List, Optional, Tuple

from src.retriever.bm25 import BM25Retriever

# 간단 전역 캐시 (프로세스 생애주기)
_CACHED_PATH: Optional[str] = None
_CACHED_IDS: Optional[List[str]] = None
_CACHED_TEXTS: Optional[List[str]] = None
_CACHED_RETRIEVER: Optional[BM25Retriever] = None


def load_corpus_tsv(path: str) -> Tuple[List[str], List[str]]:
    """Load a TSV corpus with columns: doc_id, text.

    Returns:
        doc_ids: List[str]
        texts: List[str]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    doc_ids: List[str] = []
    texts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            did = row.get("doc_id", "").strip()
            txt = row.get("text", "").strip()
            if did and txt:
                doc_ids.append(did)
                texts.append(txt)
    return doc_ids, texts


def build_retriever_from_tsv(path: str) -> Tuple[BM25Retriever, List[str], List[str]]:
    global _CACHED_PATH, _CACHED_IDS, _CACHED_TEXTS, _CACHED_RETRIEVER
    if _CACHED_PATH == path and _CACHED_RETRIEVER is not None:
        return _CACHED_RETRIEVER, _CACHED_IDS or [], _CACHED_TEXTS or []
    doc_ids, texts = load_corpus_tsv(path)
    retriever = BM25Retriever.build_from_corpus_texts(texts)
    _CACHED_PATH, _CACHED_IDS, _CACHED_TEXTS, _CACHED_RETRIEVER = path, doc_ids, texts, retriever
    return retriever, doc_ids, texts


def create_sample_corpus(output_path: str = "./resources/corpus/passages.tsv") -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = [
        "doc_id\ttext",
        "PMID:0001\tTP53 (tumor protein p53) is a tumor suppressor gene.",
        "PMID:0002\tColorectal cancer is also known as colorectal neoplasms.",
        "PMID:0003\tEGFR is implicated in multiple cancers including lung cancer.",
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
    print(f"샘플 코퍼스 생성 완료: {output_path}")


