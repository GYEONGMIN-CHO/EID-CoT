from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass
class ProjectConfig:
    openai_api_key: Optional[str] = None
    normalizer: Optional[str] = None
    scispacy_model: Optional[str] = None
    dict_mesh_tsv: Optional[str] = None
    dict_hgnc_tsv: Optional[str] = None
    dict_ncbigene_tsv: Optional[str] = None
    dict_extra_synonyms_tsv: Optional[str] = None
    corpus_tsv: Optional[str] = None
    evidence_top_k: int = 3
    link_min_score: float = 0.0
    link_min_margin: float = 0.0
    # 재랭크/캐시 옵션
    embed_rerank: bool = False
    embed_model: Optional[str] = None
    link_cache_size: int = 2048


def load_config(env_path: Optional[str] = None) -> ProjectConfig:
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()
    return ProjectConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        normalizer=os.getenv("NORMALIZER", "scispacy_biosyn"),
        scispacy_model=os.getenv("SCISPACY_MODEL", "en_ner_bc5cdr_md"),
        dict_mesh_tsv=os.getenv("DICT_MESH_TSV", "./resources/dicts/mesh.tsv"),
        dict_hgnc_tsv=os.getenv("DICT_HGNC_TSV", "./resources/dicts/hgnc.tsv"),
        dict_ncbigene_tsv=os.getenv("DICT_NCBIGENE_TSV", "./resources/dicts/ncbigene.tsv"),
        dict_extra_synonyms_tsv=os.getenv("DICT_EXTRA_SYNONYMS_TSV", "./resources/dicts/extra_synonyms.tsv"),
        corpus_tsv=os.getenv("CORPUS_TSV", "./resources/corpus/passages.tsv"),
        evidence_top_k=int(os.getenv("EVIDENCE_TOP_K", "3")),
        link_min_score=float(os.getenv("LINK_MIN_SCORE", "0.0")),
        link_min_margin=float(os.getenv("LINK_MIN_MARGIN", "0.0")),
        embed_rerank=os.getenv("EMBED_RERANK", "0") not in ("0", "false", "False"),
        embed_model=os.getenv("EMBED_MODEL"),
        link_cache_size=int(os.getenv("LINK_CACHE_SIZE", "2048")),
    )


