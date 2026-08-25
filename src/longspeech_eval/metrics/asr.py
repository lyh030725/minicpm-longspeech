from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

from jiwer import cer, wer

CJK_LANGUAGES = {"zh", "ja", "ko"}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def score_asr(rows: list[dict]) -> dict:
    per_language: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        ref = normalize_text(str(row["reference"]))
        hyp = normalize_text(str(row["prediction"]))
        per_language[str(row.get("language", "unknown"))].append((ref, hyp))

    language_scores = {}
    weighted_numerator = 0.0
    weighted_denominator = 0

    for language, pairs in sorted(per_language.items()):
        refs = [r for r, _ in pairs]
        hyps = [h for _, h in pairs]
        use_cer = language.lower() in CJK_LANGUAGES
        value = cer(refs, hyps) if use_cer else wer(refs, hyps)

        units = sum(
            len(ref.replace(" ", "")) if use_cer else max(1, len(ref.split()))
            for ref in refs
        )
        weighted_numerator += value * units
        weighted_denominator += units

        language_scores[language] = {
            "metric": "CER" if use_cer else "WER",
            "value": value,
            "samples": len(pairs),
            "reference_units": units,
        }

    return {
        "task": "ASR",
        "samples": len(rows),
        "overall_error_rate": (
            weighted_numerator / weighted_denominator if weighted_denominator else None
        ),
        "by_language": language_scores,
    }
