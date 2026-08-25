from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .io import append_jsonl, read_jsonl, write_json

JUDGE_SYSTEM_PROMPT = """You are evaluating an answer to a temporal localization question.
Compare the model answer with the reference answer and judge semantic correctness.

Return exactly one JSON object:
{"judgment":"YES|PARTIALLY|NO","reason":"brief reason"}

Definitions:
- YES: fully correct in the information and temporal relation/location requested.
- PARTIALLY: contains meaningful correct information but is incomplete, imprecise, or partly wrong.
- NO: incorrect, contradictory, irrelevant, or fails to answer the requested temporal information.
Do not reward extra unsupported claims.
"""


def _parse_judgment(text: str) -> str:
    try:
        obj = json.loads(text)
        value = str(obj.get("judgment", "")).upper()
        if value in {"YES", "PARTIALLY", "NO"}:
            return value
    except json.JSONDecodeError:
        pass

    match = re.search(r"\b(YES|PARTIALLY|NO)\b", text.upper())
    if not match:
        raise ValueError(f"Judge output did not contain YES/PARTIALLY/NO: {text!r}")
    return match.group(1)


def judge_temporal_qa(
    *,
    predictions_path: Path,
    output_path: Path,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    resume: bool = True,
) -> dict:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Temporal QA judging needs the optional dependency: uv sync --extra judge"
        ) from exc

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY or pass --api-key for temporal QA judging.")

    client = OpenAI(api_key=key, base_url=base_url)
    rows = [row for row in read_jsonl(predictions_path) if not row.get("error")]

    completed: dict[str, dict] = {}
    if resume and output_path.exists():
        completed = {
            str(row["sample_id"]): row
            for row in read_jsonl(output_path)
            if row.get("sample_id") and row.get("judgment")
        }

    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in completed:
            continue

        user_prompt = (
            f"Question/instruction:\n{row['prompt']}\n\n"
            f"Reference answer:\n{row['reference']}\n\n"
            f"Model answer:\n{row['prediction']}"
        )
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
        judgment = _parse_judgment(raw)
        judged = {
            "sample_id": sample_id,
            "judgment": judgment,
            "judge_model": model,
            "raw_judge_output": raw,
        }
        append_jsonl(output_path, judged)
        completed[sample_id] = judged

    counts = {"YES": 0, "PARTIALLY": 0, "NO": 0}
    for row in completed.values():
        value = str(row["judgment"]).upper()
        if value in counts:
            counts[value] += 1

    n = sum(counts.values())
    metrics = {
        "task": "Temporal_Relative_QA",
        "judge_model": model,
        "samples": n,
        "counts": counts,
        "strict_accuracy": counts["YES"] / n if n else None,
        "relaxed_accuracy": (counts["YES"] + counts["PARTIALLY"]) / n if n else None,
    }
    write_json(output_path.with_suffix(".metrics.json"), metrics)
    return metrics
