from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

from longspeech_eval.config import SUPPORTED_TASKS
from longspeech_eval.dataset import DEFAULT_DATASET_REPO, ensure_task_jsonl, iter_samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download selected LongSpeech task metadata and referenced WAV files."
    )
    parser.add_argument("--task", action="append", choices=SUPPORTED_TASKS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-root", type=Path, default=Path("data/longspeech"))
    parser.add_argument("--repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="download referenced WAV files too (default: true)",
    )
    args = parser.parse_args()

    tasks = args.task or list(SUPPORTED_TASKS)
    for task in tasks:
        metadata = ensure_task_jsonl(
            task=task,
            split=args.split,
            data_root=args.data_root,
            repo_id=args.repo,
            revision=args.revision,
        )
        print(f"{task}: metadata -> {metadata}")

        if not args.audio:
            continue

        samples = list(
            iter_samples(
                task=task,
                split=args.split,
                data_root=args.data_root,
                repo_id=args.repo,
                revision=args.revision,
                limit=args.limit,
            )
        )
        for sample in tqdm(samples, desc=f"download {task}", unit="audio"):
            hf_hub_download(
                repo_id=args.repo,
                repo_type="dataset",
                filename=sample.audio_relpath,
                revision=args.revision,
            )


if __name__ == "__main__":
    main()
