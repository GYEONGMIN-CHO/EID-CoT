from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

from .steplog import Evidence, Step, StepLog


def save_steplog_json(path: str | Path, steplog: StepLog) -> None:
    p = Path(path)
    p.write_text(json.dumps(asdict(steplog), ensure_ascii=False, indent=2), encoding="utf-8")


def load_steplog_json(path: str | Path) -> StepLog:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    steps: List[Step] = []
    for s in data["steps"]:
        evs = [Evidence(**e) for e in s.get("evidence", [])]
        steps.append(Step(idx=s["idx"], ids=list(s.get("ids", [])), evidence=evs, claim=s.get("claim")))
    return StepLog(qid=data["qid"], steps=steps, final_answer=data.get("final_answer", ""), citations=list(data.get("citations", [])))


def save_steplogs_jsonl(path: str | Path, logs: Iterable[StepLog]) -> None:
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        for log in logs:
            f.write(json.dumps(asdict(log), ensure_ascii=False) + "\n")


