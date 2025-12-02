from __future__ import annotations

import re

_ID_RE = re.compile(r"^(mesh:D\d{6}|HGNC:\d+|GeneID:\d+|DB\d{5})$")


def is_valid_id_text(text: str) -> bool:
    return _ID_RE.match(text) is not None


