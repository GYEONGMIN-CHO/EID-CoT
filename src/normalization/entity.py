from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Entity:
    """정규화된 엔터티 정보"""
    surface: str  # 표면형 (원문에서 추출된 텍스트)
    label: str    # 엔터티 타입 라벨
    start: int    # 시작 위치
    end: int      # 끝 위치
    norm_id: Optional[str] = None  # 정규화된 ID (예: mesh:D003110, GeneID:7157)
