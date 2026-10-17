#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${HF_PYTHON:-/home/archogenesis/.hf-cli/venv/bin/python}"
exec systemd-run --user --unit=deepseek41-exl3-download \
  -p Nice=10 -p IOWeight=20 -p MemoryMax=8G -p MemorySwapMax=0 \
  --working-directory="$PWD" \
  /usr/bin/env CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=void \
  HF_HUB_DISABLE_XET=0 HF_XET_HIGH_PERFORMANCE=0 \
  HF_XET_NUM_CONCURRENT_RANGE_GETS=2 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
  HF_HUB_DISABLE_PROGRESS_BARS=1 \
  "$python_bin" "$PWD/scripts/stage_worker.py"
