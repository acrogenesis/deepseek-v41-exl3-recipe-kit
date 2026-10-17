#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results work/no-hooks
python -m unittest discover -s tests -v > results/cpu-tests.txt 2>&1
python -m compileall -q runtime scripts tests
podman --hooks-dir "$PWD/work/no-hooks" run --rm --network none \
  --memory=4g --memory-swap=4g \
  -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void -e PYTHONPATH=/recipe \
  -v "$PWD:/recipe:ro" -w /recipe \
  localhost/deepseek41-exl3-5090:staged tests/native_metadata.py \
  > results/native-metadata-test.txt 2>&1
cat results/cpu-tests.txt results/native-metadata-test.txt
