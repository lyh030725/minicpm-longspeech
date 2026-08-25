from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

from longspeech_eval.config import DEFAULT_EVAL_LIMITS, SUPPORTED_TASKS
from longspeech_eval.dataset import DEFAULT_DATASET_REPO, ensure_task_jsonl, iter_samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the LongSpeech evaluation subset and referenced WAV files."
    )
    parser.add_argument("--task", action="append", choices=SUPPORTED_TASKS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-root", type=Path, default=Path("data/longspeech"))
    parser.add_argument("--repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--limit",
        type=int,
        help="override the default sample count for every selected task",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="download the full split for every selected task, including full ASR",
    )
    parser.add_argument(
        "--audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="download referenced WAV files too (default: true)",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be > 0")
    if args.all_samples and args.limit is not None:
        raise SystemExit("Use either --all-samples or --limit, not both.")

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

        if args.all_samples:
            effective_limit = None
        elif args.limit is not None:
            effective_limit = args.limit
        else:
            effective_limit = DEFAULT_EVAL_LIMITS[task]

        scope = "all" if effective_limit is None else f"first {effective_limit}"
        print(f"{task}: downloading {scope} referenced audio samples")

        samples = iter_samples(
            task=task,
            split=args.split,
            data_root=args.data_root,
            repo_id=args.repo,
            revision=args.revision,
            limit=effective_limit,
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
