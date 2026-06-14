#!/bin/bash
# Portable environment setup for SentinelAI.
#
# Same script on every host: detects an NVIDIA GPU and installs the CUDA build of
# PyTorch + relies on system ffmpeg's NVDEC; otherwise installs CPU PyTorch and
# falls back to software decode. Isolation is a uv-managed venv at ./.venv.
#
# Usage:   scripts/setup-env.sh
# Env:     CUDA_INDEX  override the CUDA wheel index (default: cu124)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# --- uv (fast, reproducible venv + pip) -----------------------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
echo ">> uv $(uv --version)"

# --- system ffmpeg (provides NVDEC via the driver when a GPU is present) ----
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo ">> installing ffmpeg (apt) ..."
  sudo apt-get update -qq
  sudo apt-get install -y -qq ffmpeg
fi
echo ">> ffmpeg $(ffmpeg -hide_banner -version | head -1)"

# --- isolated venv ---------------------------------------------------------
echo ">> (re)creating venv at .venv (python 3.10+) ..."
uv venv --python ">=3.10" .venv

# --- GPU-aware PyTorch -----------------------------------------------------
CUDA_INDEX="${CUDA_INDEX:-cu124}"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo ">> NVIDIA GPU detected — installing CUDA ($CUDA_INDEX) PyTorch ..."
  TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_INDEX}"
else
  echo ">> no NVIDIA GPU — installing CPU PyTorch ..."
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi
uv pip install --python .venv \
  --index-url "$TORCH_INDEX" \
  torch torchvision torchaudio

# --- project + base deps ---------------------------------------------------
echo ">> installing project (editable) + base deps ..."
uv pip install --python .venv -e .

echo
echo ">> done. Verify with:"
echo "     .venv/bin/python -m sentinelai.doctor"
