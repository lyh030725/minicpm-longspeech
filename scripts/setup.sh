#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv python install 3.10
uv sync

echo
echo "Environment ready."
echo "Smoke test:"
echo "  uv run python run.py --task ASR --limit 1"
