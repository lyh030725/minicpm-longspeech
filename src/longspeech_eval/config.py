from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_TASKS = ("ASR", "summary", "Temporal_Relative_QA")
DEFAULT_MAX_NEW_TOKENS = {
    "ASR": 8192,
    "summary": 1024,
    "Temporal_Relative_QA": 256,
}

# Evaluation subset requested for this project:
# - ASR: first 1,000 samples from the test JSONL
# - summary / Temporal Relative QA: entire test split
DEFAULT_EVAL_LIMITS: dict[str, int | None] = {
    "ASR": 1000,
    "summary": None,
    "Temporal_Relative_QA": None,
}


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "openbmb/MiniCPM-o-4_5"
    attn_implementation: str = "sdpa"
    dtype: str = "bfloat16"
    device: str = "cuda"


@dataclass(frozen=True)
class StreamingConfig:
    sample_rate: int = 16_000
    chunk_seconds: float = 1.0
    prompt_position: str = "before"

    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_seconds)
