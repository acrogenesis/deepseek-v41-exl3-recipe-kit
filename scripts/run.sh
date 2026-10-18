#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" != --gpu-is-free ]]; then
  echo 'GPU work is on hold. When free: bash scripts/run.sh --gpu-is-free smoke' >&2
  exit 2
fi
shift
mode="${1:-smoke}"
if (( $# )); then shift; fi
entry=runtime.runner
case "$mode" in
  smoke|bench) runner_args=("$mode") ;;
  diagnose) entry=runtime.diagnostic; runner_args=() ;;
  quality) entry=runtime.quality; runner_args=() ;;
  *) echo 'Use smoke, bench, diagnose or quality' >&2; exit 2;;
esac
python scripts/preflight.py
expected="$(python -c 'import json; print(json.load(open("results/runtime-image.json"))[0]["Id"])')"
test "$(podman image inspect localhost/deepseek41-exl3-5090:staged --format '{{.Id}}')" = "$expected"
mkdir -p work/kernel-cache
exec podman --hooks-dir "$PWD/work/no-hooks" run --rm --name deepseek41-exl3-5090 \
  --network none --device nvidia.com/gpu=all \
  --memory=96g --memory-swap=96g --shm-size=2g \
  -e CUDA_VISIBLE_DEVICES=0 -e NVIDIA_VISIBLE_DEVICES=void \
  -e EXL3_ATS_MMAP=0 -e EXL3_ENGRAM_ATS=0 -e EXL3_INT8_GEMV=0 \
  -e EXLLAMAV3_TUNE_CACHE=/kernel-cache/coop_autotune_v1.bin \
  -e TRITON_CACHE_DIR=/kernel-cache/triton \
  -v "$PWD/work/kernel-cache:/kernel-cache:rw" \
  -v "$PWD:/recipe:ro" -v "$PWD/results:/recipe/results:rw" -w /recipe \
  localhost/deepseek41-exl3-5090:staged -m "$entry" "${runner_args[@]}" --allow-gpu "$@"
