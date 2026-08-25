#!/usr/bin/env bash
set -euo pipefail

uv run python run.py --task ASR --resume "$@"
uv run python run.py --task summary --resume "$@"
uv run python run.py --task Temporal_Relative_QA --resume "$@"
