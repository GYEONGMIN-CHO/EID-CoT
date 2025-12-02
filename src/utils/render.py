from __future__ import annotations

from typing import Dict, List

from src.normalization.dictionaries import DictionaryLoader

from .steplog import StepLog


def render_final_answer(steplog: StepLog, id2surface: Dict[str, str] | None = None) -> str:
    id2surface = id2surface or {}
    parts: List[str] = []
    for step in steplog.steps:
        surface = ", ".join(id2surface.get(i, i) for i in step.ids)
        parts.append(surface)
    return "; ".join(parts)


def build_id2label_from_dicts() -> Dict[str, str]:
    """사전에서 ID→대표 라벨 매핑을 구축합니다."""
    loader = DictionaryLoader()
    mapping: Dict[str, str] = {}
    for entries in (loader.load_mesh(), loader.load_hgnc(), loader.load_ncbigene()):
        for entry in entries.values():
            mapping[entry.id] = entry.preferred_label
    return mapping


def render_user_friendly(steplog: StepLog, original_request: str | None = None, id2label: Dict[str, str] | None = None) -> str:
    """Render a 3-phase human-friendly log: request → reasoning → result.

    - Request: Original question/text
    - Reasoning: ID-only steps with brief evidence snippets
    - Result: Final IDs and labels
    """
    id2label = id2label or {}
    lines: List[str] = []
    # 1) Request
    if original_request:
        lines.append("[Request]")
        lines.append(original_request)
        lines.append("")
    # 2) Reasoning
    lines.append("[Reasoning]")
    if not steplog.steps:
        lines.append("(no steps)")
    else:
        for step in steplog.steps:
            ids_txt = ", ".join(step.ids) if step.ids else "(no ids)"
            lines.append(f"Step {step.idx}: {ids_txt}")
            # evidence snippets (truncate)
            for ev in step.evidence[:5]:
                snippet = ev.span
                if len(snippet) > 120:
                    snippet = snippet[:117] + "..."
                lines.append(f"  - [{ev.doc_id}] {snippet}")
    lines.append("")
    # 3) Result
    lines.append("[Result]")
    final_ids = []
    for s in steplog.steps:
        final_ids.extend(s.ids)
    labels = [id2label.get(i, i) for i in final_ids]
    lines.append("IDs: " + (", ".join(final_ids) if final_ids else "(none)"))
    lines.append("Labels: " + (", ".join(labels) if labels else "(none)"))
    return "\n".join(lines)


