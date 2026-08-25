#!/usr/bin/env bash
set -euo pipefail

EXPECTED_IMAGE="runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404-cluster"

echo "Target RunPod image: ${EXPECTED_IMAGE}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# MiniCPM-o 4.5 streaming inference is tested on Python 3.10.
uv python install 3.10

# pyproject.toml pins torch/torchaudio to PyTorch's cu128 index.
uv sync

uv run python - <<'PY'
import sys

import torch
import torchaudio
import transformers

print(f"Python       : {sys.version.split()[0]}")
print(f"PyTorch      : {torch.__version__}")
print(f"Torch CUDA   : {torch.version.cuda}")
print(f"Torchaudio   : {torchaudio.__version__}")
print(f"Transformers : {transformers.__version__}")
print(f"CUDA usable  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU           : {torch.cuda.get_device_name(0)}")

if sys.version_info[:2] != (3, 10):
    raise SystemExit("Expected Python 3.10")
if not torch.__version__.startswith("2.8.0"):
    raise SystemExit(f"Expected torch 2.8.0, got {torch.__version__}")
if torch.version.cuda != "12.8":
    raise SystemExit(f"Expected a CUDA 12.8 PyTorch build, got {torch.version.cuda}")
if not torchaudio.__version__.startswith("2.8.0"):
    raise SystemExit(f"Expected torchaudio 2.8.0, got {torchaudio.__version__}")
if transformers.__version__ != "4.51.0":
    raise SystemExit(f"Expected transformers 4.51.0, got {transformers.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access the NVIDIA GPU")
PY

echo
echo "Environment ready."
echo "Smoke test:"
echo "  uv run python run.py --task ASR --limit 1"
