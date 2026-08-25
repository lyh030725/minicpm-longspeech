from __future__ import annotations

from rouge_score import rouge_scorer


def score_summary(rows: list[dict]) -> dict:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    for row in rows:
        scores = scorer.score(str(row["reference"]), str(row["prediction"]))
        for key in totals:
            totals[key] += scores[key].fmeasure

    n = len(rows)
    return {
        "task": "summary",
        "samples": n,
        "rouge1_f1": totals["rouge1"] / n if n else None,
        "rouge2_f1": totals["rouge2"] / n if n else None,
        "rougeL_f1": totals["rougeL"] / n if n else None,
    }
