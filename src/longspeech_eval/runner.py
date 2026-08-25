from __future__ import annotations

import gc
from pathlib import Path

from tqdm import tqdm

from .config import ModelConfig, StreamingConfig
from .dataset import AudioResolver, iter_samples
from .io import append_jsonl, completed_sample_ids, read_jsonl, successful_rows, write_json
from .metrics import score_asr, score_summary
from .model import MiniCPMOStreamingModel
from .streaming import StreamingEvaluator


def evaluate(
    *,
    task: str,
    split: str,
    data_root: Path,
    output_path: Path,
    dataset_repo: str,
    dataset_revision: str,
    model_config: ModelConfig,
    streaming_config: StreamingConfig,
    max_new_tokens: int,
    start_index: int,
    end_index: int | None,
    limit: int | None,
    resume: bool,
) -> None:
    done = completed_sample_ids(output_path) if resume else set()
    samples = list(
        iter_samples(
            task=task,
            split=split,
            data_root=data_root,
            repo_id=dataset_repo,
            revision=dataset_revision,
            start_index=start_index,
            end_index=end_index,
            limit=limit,
        )
    )

    model = MiniCPMOStreamingModel(model_config)
    evaluator = StreamingEvaluator(
        model=model,
        config=streaming_config,
        max_new_tokens=max_new_tokens,
    )
    resolver = AudioResolver(
        data_root=data_root,
        repo_id=dataset_repo,
        revision=dataset_revision,
    )

    for sample in tqdm(samples, desc=task, unit="sample"):
        if sample.sample_id in done:
            continue

        audio_path = resolver.resolve(sample.audio_relpath)
        result = evaluator.run_sample(sample, audio_path)
        append_jsonl(output_path, result.to_dict())
        gc.collect()

    rows = successful_rows(read_jsonl(output_path))
    if task == "ASR":
        write_json(output_path.with_suffix(".metrics.json"), score_asr(rows))
    elif task == "summary":
        write_json(output_path.with_suffix(".metrics.json"), score_summary(rows))
