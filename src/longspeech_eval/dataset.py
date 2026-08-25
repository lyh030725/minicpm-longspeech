from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from huggingface_hub import hf_hub_download

from .config import SUPPORTED_TASKS

DEFAULT_DATASET_REPO = "ATH-MaaS/Marco_Longspeech"


@dataclass(frozen=True)
class LongSpeechSample:
    index: int
    task: str
    language: str
    audio_relpath: str
    prompt: str
    reference: str

    @property
    def sample_id(self) -> str:
        return f"{self.task}:{self.index:06d}"


def _task_jsonl_name(task: str, split: str) -> str:
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}. Expected one of {SUPPORTED_TASKS}")
    return f"LongSpeechQA/{task}/{split}.jsonl"


def ensure_task_jsonl(
    *,
    task: str,
    split: str,
    data_root: Path,
    repo_id: str = DEFAULT_DATASET_REPO,
    revision: str = "main",
) -> Path:
    local = data_root / _task_jsonl_name(task, split)
    if local.exists():
        return local

    local.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=_task_jsonl_name(task, split),
        revision=revision,
    )
    local.write_bytes(Path(cached).read_bytes())
    return local


def iter_samples(
    *,
    task: str,
    split: str,
    data_root: Path,
    repo_id: str = DEFAULT_DATASET_REPO,
    revision: str = "main",
    start_index: int = 0,
    end_index: int | None = None,
    limit: int | None = None,
) -> Iterator[LongSpeechSample]:
    path = ensure_task_jsonl(
        task=task,
        split=split,
        data_root=data_root,
        repo_id=repo_id,
        revision=revision,
    )

    yielded = 0
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index < start_index:
                continue
            if end_index is not None and index >= end_index:
                break
            if limit is not None and yielded >= limit:
                break
            if not line.strip():
                continue

            row = json.loads(line)
            messages = row.get("messages") or []
            if len(messages) < 2:
                raise ValueError(f"Malformed sample at {path}:{index + 1}: expected >=2 messages")

            user = messages[0]
            assistant = messages[-1]
            audio = user.get("audio")
            prompt = user.get("content")
            reference = assistant.get("content")

            if not isinstance(audio, str) or not audio:
                raise ValueError(f"Missing audio path at {path}:{index + 1}")
            if not isinstance(prompt, str):
                raise ValueError(f"Missing user prompt at {path}:{index + 1}")
            if not isinstance(reference, str):
                raise ValueError(f"Missing reference at {path}:{index + 1}")

            yield LongSpeechSample(
                index=index,
                task=row.get("task", task),
                language=row.get("language", "unknown"),
                audio_relpath=audio,
                prompt=prompt,
                reference=reference,
            )
            yielded += 1


class AudioResolver:
    def __init__(
        self,
        *,
        data_root: Path,
        repo_id: str = DEFAULT_DATASET_REPO,
        revision: str = "main",
    ) -> None:
        self.data_root = data_root
        self.repo_id = repo_id
        self.revision = revision

    def resolve(self, audio_relpath: str) -> Path:
        local = self.data_root / audio_relpath
        if local.exists():
            return local

        cached = hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=audio_relpath,
            revision=self.revision,
        )
        return Path(cached)
