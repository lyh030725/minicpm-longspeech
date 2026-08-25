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


def _char_text(text: str) -> str:
    # CER is evaluated on lexical characters rather than whitespace placement.
    return normalize_text(text).replace(" ", "")


def score_asr(rows: list[dict]) -> dict:
    per_language: dict[str, list[tuple[str, str]]] = defaultdict(list)
    non_cjk_refs: list[str] = []
    non_cjk_hyps: list[str] = []
    cjk_refs: list[str] = []
    cjk_hyps: list[str] = []
    all_cer_refs: list[str] = []
    all_cer_hyps: list[str] = []

    for row in rows:
        language = str(row.get("language", "unknown")).lower()
        ref = normalize_text(str(row["reference"]))
        hyp = normalize_text(str(row["prediction"]))
        per_language[language].append((ref, hyp))

        all_cer_refs.append(ref.replace(" ", ""))
        all_cer_hyps.append(hyp.replace(" ", ""))
        if language in CJK_LANGUAGES:
            cjk_refs.append(ref.replace(" ", ""))
            cjk_hyps.append(hyp.replace(" ", ""))
        else:
            non_cjk_refs.append(ref)
            non_cjk_hyps.append(hyp)

    language_scores: dict[str, dict] = {}
    for language, pairs in sorted(per_language.items()):
        refs = [r for r, _ in pairs]
        hyps = [h for _, h in pairs]
        if language in CJK_LANGUAGES:
            refs = [r.replace(" ", "") for r in refs]
            hyps = [h.replace(" ", "") for h in hyps]
            value = cer(refs, hyps)
            metric = "CER"
        else:
            value = wer(refs, hyps)
            metric = "WER"

        language_scores[language] = {
            "metric": metric,
            "value": value,
            "samples": len(pairs),
        }

    return {
        "task": "ASR",
        "samples": len(rows),
        "non_cjk_samples": len(non_cjk_refs),
        "cjk_samples": len(cjk_refs),
        "non_cjk_wer": wer(non_cjk_refs, non_cjk_hyps) if non_cjk_refs else None,
        "cjk_cer": cer(cjk_refs, cjk_hyps) if cjk_refs else None,
        "overall_cer": cer(all_cer_refs, all_cer_hyps) if all_cer_refs else None,
        "by_language": language_scores,
    }
