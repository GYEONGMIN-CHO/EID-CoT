from __future__ import annotations

from typing import Dict


class Codebook:
    """Simple ID<->shortcode mapping.

    This is a minimal placeholder; in practice you'd persist and version this.
    """

    def __init__(self, id_to_code: Dict[str, str] | None = None) -> None:
        self.id_to_code = dict(id_to_code or {})
        self.code_to_id = {v: k for k, v in self.id_to_code.items()}

    def encode(self, id_value: str) -> str:
        return self.id_to_code.get(id_value, id_value)

    def decode(self, code: str) -> str:
        return self.code_to_id.get(code, code)


