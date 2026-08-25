from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_MAX_NEW_TOKENS, SUPPORTED_TASKS, ModelConfig, StreamingConfig
from .judge import judge_temporal_qa
from .runner import evaluate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MiniCPM-o 4.5 streaming evaluation on LongSpeech"
    )
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="run MiniCPM-o streaming inference")
    run.add_argument("--task", required=True, choices=SUPPORTED_TASKS)
    run.add_argument("--split", default="test")
    run.add_argument("--data-root", type=Path, default=Path("data/longspeech"))
    run.add_argument("--output-root", type=Path, default=Path("outputs"))
    run.add_argument("--dataset-repo", default="ATH-MaaS/Marco_Longspeech")
    run.add_argument("--dataset-revision", default="main")
    run.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    run.add_argument("--device", default="cuda")
    run.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    run.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "flash_attention_2"])
    run.add_argument("--sample-rate", type=int, default=16_000)
    run.add_argument("--chunk-seconds", type=float, default=1.0)
    run.add_argument("--prompt-position", choices=["before", "after"], default="before")
    run.add_argument("--max-new-tokens", type=int)
    run.add_argument("--start-index", type=int, default=0)
    run.add_argument("--end-index", type=int)
    run.add_argument("--limit", type=int)
    run.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    judge = subparsers.add_parser("judge-temporal", help="judge Temporal Relative QA outputs")
    judge.add_argument("--predictions", type=Path, required=True)
    judge.add_argument("--output", type=Path)
    judge.add_argument("--model", default="gpt-4-turbo")
    judge.add_argument("--base-url")
    judge.add_argument("--api-key")
    judge.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    return parser


def _run(args: argparse.Namespace) -> None:
    if args.chunk_seconds <= 0:
        raise SystemExit("--chunk-seconds must be > 0")
    if args.sample_rate != 16_000:
        raise SystemExit("MiniCPM-o 4.5 streaming audio input is fixed at 16 kHz.")

    output_path = args.output_root / args.task / f"{args.split}.predictions.jsonl"
    max_new_tokens = args.max_new_tokens or DEFAULT_MAX_NEW_TOKENS[args.task]

    evaluate(
        task=args.task,
        split=args.split,
        data_root=args.data_root,
        output_path=output_path,
        dataset_repo=args.dataset_repo,
        dataset_revision=args.dataset_revision,
        model_config=ModelConfig(
            model_id=args.model,
            attn_implementation=args.attn_implementation,
            dtype=args.dtype,
            device=args.device,
        ),
        streaming_config=StreamingConfig(
            sample_rate=args.sample_rate,
            chunk_seconds=args.chunk_seconds,
            prompt_position=args.prompt_position,
        ),
        max_new_tokens=max_new_tokens,
        start_index=args.start_index,
        end_index=args.end_index,
        limit=args.limit,
        resume=args.resume,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "run":
        _run(args)
        return

    if args.command == "judge-temporal":
        output = args.output or args.predictions.with_name("temporal_judgments.jsonl")
        metrics = judge_temporal_qa(
            predictions_path=args.predictions,
            output_path=output,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            resume=args.resume,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
