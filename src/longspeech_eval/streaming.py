from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import librosa
import numpy as np
import torch

from .config import StreamingConfig
from .dataset import LongSpeechSample
from .model import MiniCPMOStreamingModel


@dataclass
class InferenceResult:
    sample_id: str
    sample_index: int
    task: str
    language: str
    audio: str
    duration_seconds: float
    prompt: str
    reference: str
    prediction: str
    prompt_position: str
    sample_rate: int
    chunk_seconds: float
    num_chunks: int
    prefill_seconds: float
    generation_seconds: float
    total_seconds: float
    peak_vram_allocated_mb: float | None
    peak_vram_reserved_mb: float | None
    error: str | None

    def to_dict(self) -> dict:
        return asdict(self)


class StreamingEvaluator:
    def __init__(
        self,
        *,
        model: MiniCPMOStreamingModel,
        config: StreamingConfig,
        max_new_tokens: int,
    ) -> None:
        self.model = model
        self.config = config
        self.max_new_tokens = max_new_tokens

    def run_sample(self, sample: LongSpeechSample, audio_path: Path) -> InferenceResult:
        t0 = time.perf_counter()
        self.model.reset()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        prediction = ""
        error = None
        prefill_seconds = 0.0
        generation_seconds = 0.0
        duration_seconds = 0.0
        num_chunks = 0

        try:
            audio, _ = librosa.load(
                str(audio_path),
                sr=self.config.sample_rate,
                mono=True,
                dtype=np.float32,
            )
            duration_seconds = len(audio) / self.config.sample_rate
            num_chunks = max(1, math.ceil(len(audio) / self.config.chunk_samples))
            session_id = f"longspeech-{sample.task}-{sample.index}-{uuid4().hex[:8]}"

            p0 = time.perf_counter()
            if self.config.prompt_position == "before":
                self.model.prefill(
                    session_id=session_id,
                    content=[sample.prompt],
                    is_last_chunk=False,
                )

            for chunk_index in range(num_chunks):
                start = chunk_index * self.config.chunk_samples
                end = min((chunk_index + 1) * self.config.chunk_samples, len(audio))
                chunk = audio[start:end]

                if len(chunk) == 0:
                    chunk = np.zeros(self.config.chunk_samples, dtype=np.float32)

                is_last_audio = chunk_index == num_chunks - 1
                if is_last_audio and len(chunk) < self.config.sample_rate:
                    chunk = np.pad(
                        chunk,
                        (0, self.config.sample_rate - len(chunk)),
                        mode="constant",
                    )

                self.model.prefill(
                    session_id=session_id,
                    content=[chunk],
                    is_last_chunk=is_last_audio,
                )

            if self.config.prompt_position == "after":
                # Audio has already been flushed with is_last_chunk=True. Add the benchmark
                # instruction as the final user-side text before generation.
                self.model.prefill(
                    session_id=session_id,
                    content=[sample.prompt],
                    is_last_chunk=False,
                )
            prefill_seconds = time.perf_counter() - p0

            g0 = time.perf_counter()
            prediction = self.model.generate(
                session_id=session_id,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            generation_seconds = time.perf_counter() - g0

        except torch.OutOfMemoryError as exc:
            error = f"CUDA_OOM: {exc}"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        peak_alloc = None
        peak_reserved = None
        if torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
            peak_reserved = torch.cuda.max_memory_reserved() / 1024**2

        return InferenceResult(
            sample_id=sample.sample_id,
            sample_index=sample.index,
            task=sample.task,
            language=sample.language,
            audio=sample.audio_relpath,
            duration_seconds=duration_seconds,
            prompt=sample.prompt,
            reference=sample.reference,
            prediction=prediction,
            prompt_position=self.config.prompt_position,
            sample_rate=self.config.sample_rate,
            chunk_seconds=self.config.chunk_seconds,
            num_chunks=num_chunks,
            prefill_seconds=prefill_seconds,
            generation_seconds=generation_seconds,
            total_seconds=time.perf_counter() - t0,
            peak_vram_allocated_mb=peak_alloc,
            peak_vram_reserved_mb=peak_reserved,
            error=error,
        )
