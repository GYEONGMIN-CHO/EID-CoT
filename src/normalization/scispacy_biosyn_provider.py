from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Optional

try:
    import spacy
    from spacy.tokens import Span
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

import importlib

from src.utils.config import load_config
from src.utils.lru import LRUCache

from .dictionaries import DictionaryLoader
from .entity import Entity
from .linker_bm25 import BM25Linker


class ScispaCyBioSynProvider:
    """scispaCy NER + BM25 사전 매칭 정규화 프로바이더

    - 다중 scispaCy 모델 병합(옵션)
    - BM25 기반 사전 링킹(MeSH/HGNC/NCBIGene) + (옵션) 임베딩 재랭크
    - spaCy/scispaCy 미설치 시 사전 기반 키워드 매칭으로 폴백
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Args:
            model_name: scispaCy 모델명 (예: 'en_ner_bc5cdr_md', 'en_ner_bionlp13cg_md')
        """
        self.config = load_config()
        self.model_name = model_name or self.config.scispacy_model or 'en_ner_bc5cdr_md'

        # 다중 모델 지원: 콤마로 구분된 모델명 허용
        self.model_names: List[str] = [m.strip() for m in self.model_name.split(',') if m.strip()]

        # scispaCy 모델(들) 로드
        self.nlp_pipes: List["spacy.Language"] = self._load_models()
        
        # 사전 로더 및 링커 초기화
        self.dict_loader = DictionaryLoader()
        self.linker = None
        self._init_linker()
        
        # 캐시 설정 (LRU)
        self._link_cache = LRUCache[str, Optional[str]](capacity=self.config.link_cache_size)
        # 임베딩 재랭크 모델 (옵션)
        self._embed_model = None
        if getattr(self.config, 'embed_rerank', False):
            self._embed_model = self._maybe_load_embed_model(self.config.embed_model)
    
    def _load_models(self) -> List["spacy.Language"]:
        """scispaCy 모델(여러 개 가능) 로드"""
        if not SPACY_AVAILABLE:
            print("spaCy가 설치되지 않았습니다. 사전 기반 링킹만 사용합니다.")
            return []
            
        nlp_list: List["spacy.Language"] = []
        for name in self.model_names:
            try:
                nlp = spacy.load(name)
                print(f"scispaCy 모델 로드 완료: {name}")
                nlp_list.append(nlp)
            except OSError:
                print(f"Warning: scispaCy 모델 '{name}'을 찾을 수 없습니다.")
                print("다음 명령어로 모델을 설치하세요:")
                print(f"pip install https://s3.amazonaws.com/allenai-scispacy/models/{name}-0.5.3.tar.gz")
        if not nlp_list:
            # 기본 spaCy 모델로 fallback
            try:
                nlp = spacy.load("en_core_web_sm")
                print("기본 spaCy 모델로 fallback")
                nlp_list.append(nlp)
            except OSError:
                print("기본 spaCy 모델도 없습니다. 'python -m spacy download en_core_web_sm' 실행하세요.")
                print("scispaCy 없이 사전 기반 링킹만 사용합니다.")
        return nlp_list
    
    def _init_linker(self) -> None:
        """BM25 링커 초기화"""
        try:
            docs = self.dict_loader.build_inverted_index()
            self.linker = BM25Linker(docs)
            print(f"BM25 링커 초기화 완료: {len(docs)} 문서")
        except Exception as e:
            print(f"Warning: BM25 링커 초기화 실패: {e}")
            self.linker = None

    def _maybe_load_embed_model(self, model_name: Optional[str]):
        """임베딩 모델 로드 (문장 임베딩 또는 SapBERT 래퍼). 실패 시 None.
        model_name 예시: 'pritamdeka/SapBERT-from-PubMedBERT-fulltext' 혹은 sentence-transformers 모델명.
        """
        if not model_name:
            return None
        try:
            st = importlib.import_module("sentence_transformers")
            SentenceTransformer = getattr(st, "SentenceTransformer")
            m = SentenceTransformer(model_name)
            print(f"임베딩 모델 로드 완료: {model_name}")
            return m
        except Exception as e:
            print(f"임베딩 모델 로드 실패: {e}")
            return None
    
    def _normalize_surface(self, surface: str) -> str:
        """표면형 정규화 (소문자, 공백 정리)"""
        # 소문자 변환 및 공백 정리
        normalized = re.sub(r'\s+', ' ', surface.strip().lower())
        return normalized
    
    def _link_entity(self, surface: str, entity_type: Optional[str] = None) -> Optional[str]:
        """엔터티 링크 (표면형 → 정규화된 ID)"""
        if not self.linker:
            return None
        
        # 캐시 확인
        cache_key = f"{surface.lower()}:{entity_type or 'all'}"
        cached = self._link_cache.get(cache_key)
        if cached is not None:
            return cached
        
        # 정규화된 표면형으로 링크
        normalized_surface = self._normalize_surface(surface)
        
        # 타입별 링크 시도
        # BM25 원본 결과
        if entity_type:
            bm25_results = self.linker.link_with_type(normalized_surface, entity_type, k=10)
        else:
            bm25_results = self.linker.link(normalized_surface, k=10)

        # 임베딩 재랭크 (옵션) - 순서만 변경, BM25 점수 유지
        use_rerank = getattr(self.config, 'embed_rerank', False) and self._embed_model is not None
        if use_rerank:
            results = self.linker.rerank(normalized_surface, bm25_results, embed_model=self._embed_model, top_m=10)
        else:
            results = bm25_results
        
        # 결과 처리
        # 점수 기반 필터/마진 적용
        norm_id = None
        if results:
            min_score = float(getattr(self.config, 'link_min_score', 0.0) or 0.0)
            min_margin = float(getattr(self.config, 'link_min_margin', 0.0) or 0.0)
            if use_rerank:
                # 재랭크 활성화 시: 상위에서 임계값 만족하는 첫 후보 선택
                chosen = None
                for cid, cscore in results:
                    if cscore >= min_score:
                        chosen = (cid, cscore)
                        break
                if chosen is not None:
                    norm_id = chosen[0]
                else:
                    # fallback: 원본 BM25 순서 기준으로 기존 마진 규칙 적용
                    if bm25_results:
                        top_id, top_score = bm25_results[0]
                        ok = top_score >= min_score
                        if ok and len(bm25_results) > 1:
                            second_score = bm25_results[1][1]
                            if (top_score - second_score) < min_margin:
                                ok = False
                        if ok:
                            norm_id = top_id
            else:
                # 재랭크 미사용: 기존 로직 유지
                top_id, top_score = results[0]
                ok = top_score >= min_score
                if ok and len(results) > 1:
                    second_score = results[1][1]
                    if (top_score - second_score) < min_margin:
                        ok = False
                if ok:
                    norm_id = top_id
        
        # 캐시 저장
        self._link_cache.put(cache_key, norm_id)
        return norm_id
    
    def _map_spacy_label_to_type(self, spacy_label: str) -> Optional[str]:
        """spaCy 라벨을 우리 타입으로 매핑"""
        label_mapping = {
            'DISEASE': 'mesh',
            'CHEMICAL': 'mesh', 
            'GENE': 'ncbigene',
            'GENE_OR_GENE_PRODUCT': 'ncbigene',
            'ORGAN': 'mesh',
            'ORGANISM': 'mesh',
            'CELL_LINE': 'mesh',
            'CELL_TYPE': 'mesh',
            'ANATOMICAL_SYSTEM': 'mesh',
            'BIOLOGICAL_PROCESS': 'mesh',
            'MOLECULAR_FUNCTION': 'mesh',
            'CELLULAR_COMPONENT': 'mesh',
        }
        return label_mapping.get(spacy_label, None)
    
    def _prefer_span(self, a: "Span", b: "Span") -> "Span":
        """겹치는 스팬 중 더 나은 스팬 선택 규칙
        우선순위: 길이(긴 것) > 타입 매핑 가능 여부 > 라벨 우선순위
        """
        len_a = a.end_char - a.start_char
        len_b = b.end_char - b.start_char
        if len_a != len_b:
            return a if len_a > len_b else b
        type_a = self._map_spacy_label_to_type(a.label_)
        type_b = self._map_spacy_label_to_type(b.label_)
        if (type_a is not None) != (type_b is not None):
            return a if type_a is not None else b
        label_priority = {
            'GENE_OR_GENE_PRODUCT': 3,
            'GENE': 3,
            'DISEASE': 2,
            'CHEMICAL': 1,
        }
        pri_a = label_priority.get(a.label_, 0)
        pri_b = label_priority.get(b.label_, 0)
        if pri_a != pri_b:
            return a if pri_a > pri_b else b
        # 동률이면 기존 유지
        return b

    def _merge_spans(self, spans: List["Span"]) -> List["Span"]:
        """다중 모델에서 나온 스팬을 겹침 기준으로 병합"""
        if not spans:
            return []
        spans_sorted = sorted(spans, key=lambda s: (s.start_char, -(s.end_char - s.start_char)))
        merged: List["Span"] = []
        for span in spans_sorted:
            placed = False
            for i, kept in enumerate(merged):
                # 겹침 여부 판단 (char 범위)
                if not (span.end_char <= kept.start_char or span.start_char >= kept.end_char):
                    better = self._prefer_span(span, kept)
                    if better is not kept:
                        merged[i] = span
                    placed = True
                    break
            if not placed:
                merged.append(span)
        # 동일 위치/라벨 중복 제거
        unique: List["Span"] = []
        seen = set()
        for s in merged:
            key = (s.start_char, s.end_char, s.label_)
            if key in seen:
                continue
            seen.add(key)
            unique.append(s)
        return unique
    
    def annotate(self, text: str) -> List[Entity]:
        """텍스트에서 엔터티 추출 및 정규화
        
        Args:
            text: 입력 텍스트
            
        Returns:
            List[Entity]: 추출된 엔터티 리스트
        """
        entities = []
        
        if self.nlp_pipes:
            # 각 모델로 NER 수행 후 스팬 병합
            all_spans: List[Span] = []
            for nlp in self.nlp_pipes:
                doc = nlp(text)
                all_spans.extend(list(doc.ents))

            # 겹치는 스팬 병합
            merged_spans: List[Span] = self._merge_spans(all_spans)

            for span in merged_spans:
                surface = span.text
                start = span.start_char
                end = span.end_char

                entity_type = self._map_spacy_label_to_type(span.label_)
                norm_id = self._link_entity(surface, entity_type)

                entity = Entity(
                    surface=surface,
                    label=span.label_,
                    start=start,
                    end=end,
                    norm_id=norm_id
                )
                entities.append(entity)
        else:
            # scispaCy가 없으면 간단한 키워드 매칭
            entities = self._simple_keyword_matching(text)
        
        # 중복 제거 (같은 위치의 엔터티)
        unique_entities = []
        seen_positions = set()
        
        for entity in entities:
            position = (entity.start, entity.end)
            if position not in seen_positions:
                seen_positions.add(position)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _simple_keyword_matching(self, text: str) -> List[Entity]:
        """간단한 키워드 매칭 (scispaCy 없이)"""
        entities = []
        
        # 사전에서 모든 표면형을 수집
        all_surfaces = set()
        mesh_entries = self.dict_loader.load_mesh()
        hgnc_entries = self.dict_loader.load_hgnc()
        ncbigene_entries = self.dict_loader.load_ncbigene()
        
        for entry in mesh_entries.values():
            all_surfaces.update(entry.all_labels())
        for entry in hgnc_entries.values():
            all_surfaces.update(entry.all_labels())
        for entry in ncbigene_entries.values():
            all_surfaces.update(entry.all_labels())
        
        # 텍스트에서 매칭
        text_lower = text.lower()
        for surface in all_surfaces:
            surface_lower = surface.lower()
            if surface_lower in text_lower:
                start = text_lower.find(surface_lower)
                end = start + len(surface_lower)
                
                # 정규화된 ID 링크
                norm_id = self._link_entity(surface, None)
                
                entity = Entity(
                    surface=surface,
                    label="UNKNOWN",
                    start=start,
                    end=end,
                    norm_id=norm_id
                )
                entities.append(entity)
        
        return entities
    
    def get_stats(self) -> dict:
        """프로바이더 통계 정보"""
        stats = {
            'model_names': self.model_names,
            'linker_stats': self.linker.get_stats() if self.linker else None,
            'cache_size': len(self._link_cache)
        }
        return stats


def test_scispacy_provider():
    """scispaCy 프로바이더 테스트"""
    try:
        provider = ScispaCyBioSynProvider()
        
        # 테스트 텍스트
        test_text = "TP53 is frequently mutated in colon cancer and breast cancer."
        
        print("=== scispaCy 프로바이더 테스트 ===")
        print(f"입력: {test_text}")
        
        # 엔터티 추출
        entities = provider.annotate(test_text)
        
        print(f"\n추출된 엔터티 ({len(entities)}개):")
        for entity in entities:
            print(f"  {entity.surface} [{entity.label}] @({entity.start},{entity.end}) -> {entity.norm_id}")
        
        # 통계
        stats = provider.get_stats()
        print(f"\n프로바이더 통계: {stats}")
        
    except Exception as e:
        print(f"테스트 실패: {e}")
        print("scispaCy 모델이 설치되지 않았을 수 있습니다.")


if __name__ == "__main__":
    test_scispacy_provider()
