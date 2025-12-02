from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, corpus_tokens: Sequence[Sequence[str]]) -> None:
        self.bm25 = BM25Okapi(corpus_tokens)

    def search(self, query_tokens: Iterable[str], k: int = 5) -> List[Tuple[int, float]]:
        scores = self.bm25.get_scores(list(query_tokens))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    @staticmethod
    def build_from_corpus_texts(corpus: Sequence[str]) -> "BM25Retriever":
        from src.utils.tokenize import tokenize_corpus
        return BM25Retriever(tokenize_corpus(corpus))


