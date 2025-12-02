# EID-CoT: Entity ID-centric Chain-of-Thought

**바이오메디컬 추론의 모호성 제거 및 근거성 향상을 위한 정규화 엔터티 ID 기반 CoT**

EID-CoT는 LLM의 중간 추론 과정을 자연어가 아닌 **표준화된 생의학 엔터티 ID(MeSH, HGNC, GeneID)**로 강제하여, 동음이의어/동의어로 인한 환각(Hallucination)을 구조적으로 차단하고 근거 기반 추론(Grounding)을 강화하는 프로젝트입니다.

---

## 🚀 주요 기능

- **ID-only Reasoning**: 추론의 중간 단계를 표준 ID로만 구성하여 개념의 모호성 제거
- **Robust Normalization**: scispaCy 다중 모델 병합 + BM25 사전 링킹을 통한 강력한 엔터티 정규화
- **Evidence Retrieval**: ID 기반 쿼리 변형을 통해 외부 코퍼스에서 정확한 근거 문서 수집
- **StepLog System**: 추론의 각 단계를 구조화된 로그(JSON)로 저장하여 감사(Audit) 및 평가 용이
- **Automatic Evaluation**: Entity Grounding F1, Attribution, RAF 등 다양한 평가지표 자동 산출

---

## 🛠️ 설치 및 환경 설정

### 1. 환경 준비
Python 3.13 환경을 권장합니다.

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -U pip
pip install -r requirements.txt
```

### 2. 환경 변수 설정
`.env` 파일을 생성하고 필요한 설정을 추가합니다.

```bash
cp .env.example .env
```

**필수 변수:**
- `OPENAI_API_KEY`: LLM 사용을 위한 API 키

---

## ⚡ 빠른 실행 (Quick Start)

### 1. 데모 실행
단일 문장에 대해 파이프라인을 실행하고 결과를 확인합니다.

```bash
PYTHONPATH=. python experiments/run_pipeline.py "TP53 is frequently mutated in colon cancer."
```

### 2. 평가 (Mini-set)
BC5CDR 또는 NCBI 데이터셋의 미니셋을 이용해 성능을 평가합니다.

```bash
PYTHONPATH=. python experiments/run_eval_miniset.py bc5cdr
```

### 3. 문서 자동 업데이트
파이프라인 실행 결과를 문서에 자동 반영합니다.

```bash
PYTHONPATH=. python experiments/update_pipeline_md.py --md docs/step/pipeline.md
```

---

## 📚 문서 (Documentation)

더 자세한 내용은 아래 문서를 참고하세요.

- **[시스템 아키텍처 (Architecture)](docs/ARCHITECTURE.md)**: 시스템 구조, 모듈 설명, 데이터 흐름
- **[논문 초안 (Paper Draft)](docs/PAPER_DRAFT.md)**: 연구 배경, 방법론, 실험 결과

---

## 📂 디렉토리 구조

```
src/
  normalization/   # 엔터티 정규화 (NER + Linking)
  retriever/       # 증거 검색 (BM25)
  pipeline/        # ID-CoT 파이프라인
  decoding/        # 제약 디코딩
  evaluation/      # 평가 지표
  utils/           # 유틸리티
experiments/       # 실험 및 데모 스크립트
resources/         # 사전 및 코퍼스 데이터
docs/              # 프로젝트 문서
```
