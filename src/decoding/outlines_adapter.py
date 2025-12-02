from __future__ import annotations

import re
from typing import Tuple

try:
    from outlines.types import Regex as OutRegex  # type: ignore
except Exception:  # outlines optional
    OutRegex = None  # type: ignore


def build_id_output_type():  # returns OutRegex | None
    """Return Outlines Regex output type if available; otherwise None.

    Allowed patterns: mesh:D\d{6} | HGNC:\d+ | GeneID:\d+ | DB\d{5}
    """
    if OutRegex is None:
        return None
    pattern = r"(mesh:D\d{6}|HGNC:\d+|GeneID:\d+|DB\d{5})"
    return OutRegex(pattern)


def python_regex_for_ids() -> re.Pattern[str]:
    """Python `re` compiled pattern equivalent to `build_id_output_type`.

    Useful for offline validation or filtering without generation.
    """
    return re.compile(r"^(mesh:D\d{6}|HGNC:\d+|GeneID:\d+|DB\d{5})$")


def is_valid_id_text(text: str) -> bool:
    return python_regex_for_ids().match(text) is not None


