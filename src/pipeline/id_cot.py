from __future__ import annotations

from typing import List, Optional


from src.decoding.id_regex import is_valid_id_text
from src.normalization.entity import Entity
from src.normalization.provider import NormalizationProvider, build_provider
from src.retriever.bm25 import BM25Retriever
from src.retriever.corpus_utils import build_retriever_from_tsv
from src.utils.config import load_config
from src.utils.render import render_final_answer
from src.utils.steplog import Evidence, Step, StepLog
from src.utils.tokenize import tokenize


def _collect_ids(entities: List[Entity]) -> List[str]:
    seen = set()
    out: List[str] = []
    for e in entities:
        if e.norm_id and is_valid_id_text(e.norm_id) and e.norm_id not in seen:
            seen.add(e.norm_id)
            out.append(e.norm_id)
    return out


def build_steplog_from_text(qid: str, text: str, provider: NormalizationProvider | None = None, corpus_path: Optional[str] = None) -> StepLog:
    """LLM 키가 있으면 반드시 LLM 경로 사용, 없으면 오류.

    - OPENAI_API_KEY 존재 시: LangGraph 파이프라인(graph_pipeline)을 실행하여 ID 추출 및 증거 수집
    - 존재하지 않으면: RuntimeError 발생(폴백 금지)
    """
    cfg = load_config()
    import os as _os
    api_key_present = bool((cfg.openai_api_key) or _os.getenv("OPENAI_API_KEY"))
    
    if not api_key_present:
        raise RuntimeError("OPENAI_API_KEY not set (LLM path enforced)")

    # Use LangGraph Pipeline
    from src.pipeline.graph_pipeline import build_graph
    
    app = build_graph()
    initial_state = {
        "qid": qid,
        "text": text,
        "entities": [],
        "candidates": {},
        "resolved_ids": [],
        "final_answer": ""
    }
    
    # Invoke Graph
    result = app.invoke(initial_state)
    ids = result.get("resolved_ids", [])
    
    # Build Evidence (Reuse existing logic for now, or move to graph later)
    evidence = _build_evidence(text, entities=[], ids=ids, corpus_path=corpus_path)
    
    steps = [Step(idx=1, ids=ids, evidence=evidence)]
    steplog = StepLog(qid=qid, steps=steps, final_answer="", citations=["INPUT"])
    steplog.final_answer = render_final_answer(steplog)
    return steplog


def _build_evidence(text: str, entities: List[Entity], ids: Optional[List[str]] = None, corpus_path: Optional[str] = None) -> List[Evidence]:
    """입력 텍스트를 미니 코퍼스로 간주하여 증거 스팬/유사 패시지 로그화.
    실제 운영에서는 외부 코퍼스 토큰화/인덱싱으로 교체.
    """
    ev: List[Evidence] = []
    used_corpus = False

    # 1) 입력 내 스팬 위치 기록
    for e in entities:
        if e.norm_id and is_valid_id_text(e.norm_id):
            ev.append(Evidence(doc_id="INPUT", span=f"{e.surface}@{e.start}:{e.end}"))

    # 2) 외부 코퍼스가 명시적으로 제공된 경우에만 BM25로 상위 패시지 증거 추가
    import os
    if corpus_path and os.path.exists(corpus_path):
        try:
            retriever, doc_ids, texts = build_retriever_from_tsv(corpus_path)
            cfg = load_config()
            top_k = max(1, int(getattr(cfg, 'evidence_top_k', 3)))
            # 쿼리 후보 구성: (1) 엔터티 표면형 (2) ID 동의어
            from src.normalization.dictionaries import DictionaryLoader
            dl = None
            if ids:
                try:
                    dl = DictionaryLoader()
                except Exception:
                    dl = None

            def _query_variants_for_entity(e: Entity) -> List[str]:
                variants = {e.surface}
                if dl and e.norm_id and is_valid_id_text(e.norm_id):
                    entry = dl.get_entry(e.norm_id)
                    if entry:
                        variants.update(entry.all_labels())
                return list(variants)

            def _query_variants_for_id(i: str) -> List[str]:
                variants = {i}
                if dl:
                    entry = dl.get_entry(i)
                    if entry:
                        variants.update(entry.all_labels())
                return list(variants)

            # 엔터티 기반 쿼리
            for e in entities:
                if not (e.norm_id and is_valid_id_text(e.norm_id)):
                    continue
                query_list = _query_variants_for_entity(e)
                collected = 0
                seen_idx = set()
                for q in query_list:
                    if collected >= top_k:
                        break
                    qt = tokenize(q)
                    for idx, score in retriever.search(qt, k=top_k):
                        if idx in seen_idx:
                            continue
                        seen_idx.add(idx)
                        ev.append(Evidence(doc_id=doc_ids[idx], span=texts[idx]))
                        collected += 1
                        if collected >= top_k:
                            break

            # LLM ID 기반 쿼리
            for i in (ids or []):
                if not is_valid_id_text(i):
                    continue
                query_list = _query_variants_for_id(i)
                collected = 0
                seen_idx = set()
                for q in query_list:
                    if collected >= top_k:
                        break
                    qt = tokenize(q)
                    for idx, score in retriever.search(qt, k=top_k):
                        if idx in seen_idx:
                            continue
                        seen_idx.add(idx)
                        ev.append(Evidence(doc_id=doc_ids[idx], span=texts[idx]))
                        collected += 1
                        if collected >= top_k:
                            break
            used_corpus = True
        except Exception:
            # 코퍼스 로드 실패 시 입력 문장 기반으로 폴백
            pass

    if not used_corpus:
        # Better sentence splitting
        import re
        # Split by . but avoid common abbreviations (e.g., et al., vs., etc.)
        # This is a simple heuristic.
        raw_sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text.replace("\n", " "))
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        if len(sentences) > 1:
            retriever = BM25Retriever.build_from_corpus_texts(sentences)
            # 엔터티 표면형 쿼리
            for e in entities:
                if e.surface:
                    query_tokens = tokenize(e.surface)
                    top = retriever.search(query_tokens, k=1)
                    if top:
                        idx, _ = top[0]
                        ev.append(Evidence(doc_id=f"SENT_{idx}", span=sentences[idx]))
            # ID 라벨 쿼리 (사전 라벨을 문장에 매칭)
            if ids:
                try:
                    from src.normalization.dictionaries import DictionaryLoader
                    dl = DictionaryLoader()
                    labels = set()
                    for i in ids:
                        entry = dl.get_entry(i)
                        if entry:
                            labels.update(entry.all_labels())
                    for lab in list(labels)[:10]:  # 가벼운 매칭만 수행
                        qt = tokenize(lab)
                        top = retriever.search(qt, k=1)
                        if top:
                            idx, _ = top[0]
                            ev.append(Evidence(doc_id=f"SENT_{idx}", span=sentences[idx]))
                except Exception:
                    pass

    # Evidence deduplication (doc_id + normalized span)
    import re as _re
    def _norm(s: str) -> str:
        return _re.sub(r"\s+", " ", s.strip().lower())
    dedup: List[Evidence] = []
    seen_keys = set()
    for e in ev:
        key = (e.doc_id, _norm(e.span))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        dedup.append(e)
    return dedup


