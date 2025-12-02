from __future__ import annotations

import re
from typing import List, Optional, Tuple

from rank_bm25 import BM25Okapi


class BM25Linker:
    """BM25 기반 엔터티 링커"""
    
    def __init__(self, corpus_docs: List[dict]):
        """
        Args:
            corpus_docs: List[dict] with keys 'id', 'text', 'type'
        """
        self.corpus_docs = corpus_docs
        self.bm25 = None
        self._build_index()
    
    def _build_index(self) -> None:
        """BM25 인덱스 구축"""
        if not self.corpus_docs:
            print("Warning: 빈 코퍼스로 BM25 인덱스 구축")
            return
        
        # 토큰화된 문서들
        tokenized_docs = []
        for doc in self.corpus_docs:
            tokens = self._tokenize(doc['text'])
            tokenized_docs.append(tokens)
        
        self.bm25 = BM25Okapi(tokenized_docs)
        print(f"BM25 인덱스 구축 완료: {len(tokenized_docs)} 문서")
    
    def _tokenize(self, text: str) -> List[str]:
        """간단한 토큰화 (소문자, 알파벳/숫자만)"""
        # 소문자 변환 후 알파벳/숫자만 추출
        normalized = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
        tokens = normalized.split()
        return [token for token in tokens if token]
    
    def link(self, surface: str, k: int = 5) -> List[Tuple[str, float]]:
        """표면형을 사전의 ID로 링크
        
        Args:
            surface: 링크할 표면형 (예: "TP53", "colon cancer")
            k: 반환할 상위 k개 결과
            
        Returns:
            List[Tuple[str, float]]: (ID, BM25 점수) 리스트
        """
        if not self.bm25 or not self.corpus_docs:
            return []
        
        # 표면형 토큰화
        query_tokens = self._tokenize(surface)
        if not query_tokens:
            return []
        
        # BM25 점수 계산
        scores = self.bm25.get_scores(query_tokens)
        
        # 상위 k개 결과 추출
        scored_docs = list(zip(self.corpus_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc, score in scored_docs[:k]:
            if score > 0:  # 점수가 0보다 큰 경우만
                results.append((doc['id'], score))
        
        return results
    
    def link_with_type(self, surface: str, entity_type: Optional[str] = None, k: int = 5) -> List[Tuple[str, float]]:
        """특정 타입의 엔터티만 링크
        
        Args:
            surface: 링크할 표면형
            entity_type: 엔터티 타입 ('mesh', 'hgnc', 'ncbigene')
            k: 반환할 상위 k개 결과
            
        Returns:
            List[Tuple[str, float]]: (ID, BM25 점수) 리스트
        """
        if not entity_type:
            return self.link(surface, k)
        
        # 특정 타입의 문서만 필터링
        filtered_docs = [doc for doc in self.corpus_docs if doc.get('type') == entity_type]
        
        if not filtered_docs:
            return []
        
        # 필터링된 문서로 임시 링커 생성
        temp_linker = BM25Linker(filtered_docs)
        return temp_linker.link(surface, k)

    # 재랭크 훅: 외부 임베딩 기반 후보 재정렬을 허용
    def rerank(self, surface: str, candidates: List[Tuple[str, float]], embed_model: Optional[object] = None,
               top_m: int = 10) -> List[Tuple[str, float]]:
        """BM25 후보(candidates)를 임베딩 유사도로 재정렬.

        embed_model: .encode(List[str]) -> ndarray 를 지원하는 임베딩 모델 (예: sentence-transformers, SapBERT 랩퍼)
        top_m: 재랭크 시 고려할 상위 BM25 후보 수
        """
        if not candidates or embed_model is None:
            return candidates
        subset = candidates[:top_m]
        id_to_text = {doc['id']: doc['text'] for doc in self.corpus_docs}
        texts = [id_to_text.get(cid, "") for cid, _ in subset]
        # 표면형과 후보 텍스트를 임베딩 후 코사인 유사도로 정렬
        try:
            import numpy as np
            q_vec = embed_model.encode([surface])[0]
            d_mat = embed_model.encode(texts)
            # 정규화
            def l2norm(x: np.ndarray) -> np.ndarray:
                n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
                return x / n
            q = l2norm(q_vec)
            D = l2norm(d_mat)
            sims = (D @ q.reshape(-1, 1)).reshape(-1)
            # 유사도로 정렬은 하되, 반환 점수는 원래 BM25 점수를 보존
            cid_to_bm25 = {cid: bm for cid, bm in subset}
            order = [cid for cid, _ in sorted(zip([cid for cid, _ in subset], sims.tolist()), key=lambda x: x[1], reverse=True)]
            return [(cid, cid_to_bm25.get(cid, 0.0)) for cid in order]
        except Exception:
            return candidates
    
    def get_stats(self) -> dict:
        """링커 통계 정보 반환"""
        stats = {
            'total_docs': len(self.corpus_docs),
            'types': {}
        }
        
        for doc in self.corpus_docs:
            doc_type = doc.get('type', 'unknown')
            stats['types'][doc_type] = stats['types'].get(doc_type, 0) + 1
        
        return stats


def test_bm25_linker():
    """BM25 링커 테스트"""
    # 샘플 문서들
    sample_docs = [
        {'id': 'GeneID:7157', 'text': 'tp53', 'type': 'ncbigene'},
        {'id': 'HGNC:1097', 'text': 'tp53', 'type': 'hgnc'},
        {'id': 'GeneID:7157', 'text': 'p53', 'type': 'ncbigene'},
        {'id': 'HGNC:1097', 'text': 'p53', 'type': 'hgnc'},
        {'id': 'mesh:D003110', 'text': 'colon cancer', 'type': 'mesh'},
        {'id': 'mesh:D003110', 'text': 'colorectal cancer', 'type': 'mesh'},
        {'id': 'mesh:D003110', 'text': 'colorectal neoplasms', 'type': 'mesh'},
    ]
    
    linker = BM25Linker(sample_docs)
    
    print("=== BM25 링커 테스트 ===")
    
    # TP53 테스트
    results = linker.link("TP53", k=3)
    print(f"\nTP53 링크 결과:")
    for doc_id, score in results:
        print(f"  {doc_id}: {score:.4f}")
    
    # colon cancer 테스트
    results = linker.link("colon cancer", k=3)
    print(f"\ncolon cancer 링크 결과:")
    for doc_id, score in results:
        print(f"  {doc_id}: {score:.4f}")
    
    # 타입별 필터링 테스트
    results = linker.link_with_type("TP53", "ncbigene", k=2)
    print(f"\nTP53 (NCBIGene만) 링크 결과:")
    for doc_id, score in results:
        print(f"  {doc_id}: {score:.4f}")
    
    # 통계
    stats = linker.get_stats()
    print(f"\n링커 통계: {stats}")


if __name__ == "__main__":
    test_bm25_linker()
