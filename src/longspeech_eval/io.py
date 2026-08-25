from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def completed_sample_ids(path: Path) -> set[str]:
    return {str(row["sample_id"]) for row in read_jsonl(path) if row.get("sample_id")}


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def successful_rows(rows: Iterable[dict]) -> list[dict]:
    return [row for row in rows if not row.get("error")]
