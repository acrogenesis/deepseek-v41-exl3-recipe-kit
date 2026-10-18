# Grouped expert dispatch experiments — 2026-10-18 UTC

Goal remains above 30 output tokens/s on the same EXL3 checkpoint, one sequence.
That goal has not been reached. Model, quantization, Engram tables, selected
experts, context configuration, prompts and sampling match the previous
[target30 campaign](../20261018-target30/METHOD.md).

All changes are opt-in. Default execution remains the registered-RAM LRU path.
No external transfer or second model process ran alongside these measurements.
Full suites use eight exact checks and three 128-token prompts, fresh then repeat,
with fresh KV and recurrent state per request. Context is configured to 8192;
these short prompts do not fill it. Timings exclude the first output token.
Individual sweeps, not confidence intervals. See summary.json for final numbers.

## What was measured

1. Torch CPU/CUDA profile of eight decode steps after warm-up. The trace records
   about 5200 GPU kernels/token, 23.44 ms kernel time/token and 14.26 ms memcpy
   time/token. These are summed device activities, not independently additive
   wall-clock components; overlap is possible. Instrumentation slows execution.
   The 115 MB Chrome trace remains local and ignored.
2. A native C++ loop preserving each ATen operation: no useful speed improvement,
   all 64 x 129280 teacher-forced scores unchanged. This differs from the earlier
   rejected native fused activation, whose clamp ordering changed arithmetic.
3. Grouped K2/K3 MUL1 GEMV: independent expert/column work runs in one cooperative
   grid. Gate and up projections share one launch; down projections use another.
   The existing CFG 0 trellis decoding, MMA fold cadence, FP32 warp reduction and
   input/output Hadamard operations are preserved. ATen activation and ordered
   FP32 weighted sums retain the previous semantics. Prefill keeps the eager path.
4. CUDA graph replay of each unchanged single-row hyperconnection mixer.
   Attention, shared experts and routing are unchanged. This consumes additional
   GPU graph memory and had inconsistent full-suite timings; it stays optional.
5. LFU GPU eviction, with LRU ties. Access counts depend only on already observed
   requests, including prefill. It can retain established hot experts across
   prompts but can slow a new topic. RAM remains LRU. It changes residency only;
   all 384 experts/layer remain routable.

## Validation and provenance

`grouped-unit.json` compares real K2/K3 weights at input scales 0.1, 1 and 10,
including uncapped and capped activation, broadcast and per-expert input rows.
All 72 initial cases are bit-identical. Cached expert compute is about 100–110 μs
versus 310–339 μs in that microbenchmark; this excludes paging and the rest of the
model. The later grouped-reuse.json passes all 78 cases, including forced GPU slot
eviction and reuse. LFU eviction also passes 24 real-byte DMA/lifetime cases.

`grouped-diagnostic/` and `hc-diagnostic/` compare baseline/current/baseline
within one process. They force the same 64 continuation tokens and save every
vocabulary score. The local NPY files are ignored; summary.json records SHA256.
Warm diagnostic throughput is not the full-suite benchmark and is never used
as the headline result. Capturing and copying scores adds measurement overhead.

`full-policy-simulation.json` replays actual routes from `hc-full.json`; its
LRU/LRU miss counts match the recorded run. Simulation explores GPU and RAM
policies without reading weights or running inference. It predicts traffic,
not throughput. RAM LFU/decay code was explored but not GPU-benchmarked and was
removed from the active runtime. Its source is retained under variants/.

Baseline is commit 83812d0. ExLlama remains pinned to 954a8ca6e59d48c3e3462068ecf083fe9990f4dc.
The grouped CUDA body derives from that engine's exl3_gemv_kernel.cuh; the batched
worklist design also appears in vcruz305/vllm-exl3 at 2a740a3773601c994072295fde45738ef8cea55e.
See runtime/vendor/THIRD_PARTY_NOTICES.md for the preserved MIT notice.
The local extension links against the unchanged pinned container extension.
Build artifacts and tuning caches live in work/kernel-cache, not the model files.

## Reproduction

```sh
bash scripts/run.sh --gpu-is-free bench --grouped-dispatch \
  --output /recipe/results/grouped.json
bash scripts/run.sh --gpu-is-free bench --grouped-dispatch --gpu-policy lfu \
  --output /recipe/results/grouped-lfu.json
# Optional experiment:
bash scripts/run.sh --gpu-is-free bench --grouped-dispatch --hc-graphs \
  --output /recipe/results/grouped-hc.json
```

Keep CUDA tuning caches unchanged for score comparisons. Output files must not
already exist. Diagnostic and validation drivers are archived alongside results.
