#!/usr/bin/env bash
set -euo pipefail

# Defaults are task-aware: ASR=first 1000, summary=all, Temporal Relative QA=all.
uv run python run.py --task ASR --resume "$@"
uv run python run.py --task summary --resume "$@"
uv run python run.py --task Temporal_Relative_QA --resume "$@"
