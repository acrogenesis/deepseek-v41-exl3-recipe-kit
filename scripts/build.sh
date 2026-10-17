#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p work/no-hooks results
expected="$(python -c 'import json; print(json.load(open("runtime.lock.json"))["exllamav3"]["revision"])')"
test "$(git -C work/exllamav3 rev-parse HEAD)" = "$expected"
test -z "$(git -C work/exllamav3 status --porcelain)"
base="$(python -c 'import json; print(json.load(open("runtime.lock.json"))["base_image"]["id"])')"
test "$(podman image inspect localhost/glm53-sm120:20261001 --format '{{.Id}}')" = "$base"
# CUDA compilation is CPU work. No GPU CDI device or NVIDIA hook is passed.
podman --hooks-dir "$PWD/work/no-hooks" build --pull=never \
  --memory=12g --memory-swap=12g --cpu-shares=128 \
  -t localhost/deepseek41-exl3-5090:staged -f Containerfile .
podman image inspect localhost/deepseek41-exl3-5090:staged | python -c '
import json,sys
d=json.load(sys.stdin)[0]
print(json.dumps([{k:d[k] for k in ("Id","Created","Architecture","Os")}],indent=2))
' > results/runtime-image.json
