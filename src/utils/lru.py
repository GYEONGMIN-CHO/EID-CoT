from __future__ import annotations

from collections import OrderedDict
from typing import Generic, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """간단한 LRU 캐시 구현 (의존성 없이 OrderedDict 사용).

    - get: 키가 존재하면 값을 반환하고, 가장 최근 사용으로 갱신
    - put: 값을 설정하고, 용량 초과 시 가장 오래된 항목 제거
    """

    def __init__(self, capacity: int = 1024) -> None:
        if capacity <= 0:
            raise ValueError("LRUCache capacity must be > 0")
        self.capacity = int(capacity)
        self._store: "OrderedDict[K, V]" = OrderedDict()

    def get(self, key: K) -> Optional[V]:
        if key not in self._store:
            return None
        value = self._store.pop(key)
        self._store[key] = value  # move to end (most recently used)
        return value

    def put(self, key: K, value: V) -> None:
        if key in self._store:
            self._store.pop(key)
        self._store[key] = value
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)  # remove least recently used

    def __len__(self) -> int:
        return len(self._store)


