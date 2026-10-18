# Testing the 30 tok/s target

RTX 5090 32 GB, Ryzen 9950X3D, 128 GB RAM, Samsung 990 PRO on encrypted Btrfs.
Same 1017 EXL3 2.9 bpw weights, original Engram tables and pinned ExLlama image
as the [previous comparison](../20261018-clean-optimization/METHOD.md).
No expert pruning, routing bias, new quantization or model substitution.

## Full benchmark

Use `summary.json` and `selected-84g.json` for the selected configuration. Each run uses the
same eight exact checks followed by three 128-token speed prompts, then repeats
those prompts. All requests have fresh KV and recurrent state. Inputs contain
30/31/31 tokens; the configured context is 8,192 tokens, not a filled window.
The timer counts delivered output tokens after the first token. RAM and Engram
page caches can warm. There were no concurrent model uploads or downloads.

The previous default used 22 GiB of expert VRAM and 64 GiB of pageable RAM.
Candidates keep the same VRAM and increase the bounded RAM cache to 84 GiB.
The 96 GiB container limit, zero process swap and 12 GiB host reserve remain.
Final reports sample available host memory, cgroup usage/swap and GPU memory
after each request. These samples are not a continuous host-memory minimum.

`larger-ram.json` uses reusable I/O staging and pageable RAM. `pinned-84g.json`
registers each new RAM buffer for direct GPU transfers, but unregisters it on
every eviction. That saves a copy but adds enough registration cost to regress
the full workload. `reused-pinned-84g.json` recycles equal-size registered RAM
buffers after their outstanding DMA events finish, and also tries retaining
GPU-slot weight handles. `final-84g.json` removes the GPU-handle experiment:
that helps the short warm diagnostic, but regresses the full suite from
8.98/12.34 to 8.40/11.31 tok/s. `selected-84g.json` repeats the full suite with
GPU-handle reuse restored. Registration reuse and the single reusable O_DIRECT
staging buffer remain in both versions.

These are individual paired sweeps, not confidence intervals or broad model
quality measurements. The summary checks every output token against the prior
benchmark and verifies that the final cache/store source hashes match the run.

## Diagnostic traces

`runtime.diagnostic` forces the first speed prompt's known continuation and
records all vocabulary scores. Each trace uses 64 output positions. Timings
exclude position zero, which includes prompt prefill. Trace timings also include
score copies to CPU and Python instrumentation, so they are kept separate from
the full benchmark. Each pass recreates its GPU expert cache and generator;
the second and later passes retain a warm RAM cache.

The initial trace transfers 856.8 MB of expert weights to the GPU per decode
token. Its cold pass reads 458.7 MB/token from SSD; the immediately repeated pass
reads no expert weights from SSD. The same request touches about 59.6 GiB of
distinct experts including prefill, explaining why this short request fits the
64 GiB cache while varied prompts churn it.

Direct registered-memory DMA reaches 19.37 tok/s versus 12.23 for the staged
copy in one same-process warm diagnostic. CPU thread tests reach 13.14 with one
thread, 13.26 with four, and 19.24 with sixteen; the runtime keeps sixteen.
These are warm short-request measurements, not the headline suite rate.

The transfer microbenchmark measures about 33.6 GB/s for a pinned 13.3 MB GPU
copy. At the traced miss rate, 30 tok/s would require roughly 25.7 GB/s of H2D
traffic before accounting for compute and expert SSD misses. This is a traffic
budget, not proof of an achievable end-to-end rate. Its CPU-copy numbers reuse
the same small buffers and must not be treated as streaming DRAM bandwidth.

## Correctness and rejected candidates

- Final validation: 15 CPU tests, 24 real-byte transfer/lifetime cases including
  RAM eviction during delayed DMA and failed-read cleanup, and eight native
  resident versus paged MoE cases at K2/K3. Native comparisons retain the existing
  FP16 tolerance rather than claiming native/paged arithmetic is bit-identical.
- Every retained 64-position diagnostic has the same NPY SHA-256 across all
  129,280 vocabulary scores per position. The one native grouped-MoE experiment
  differs and is explicitly excluded. Hashes are in `summary.json`; large raw
  NPY/profile files remain local and gitignored.
- The grouped native loop changed scores and one argmax; its fused activation
  arithmetic differs. It was removed. Its source is archived in
  `native-decode/experimental-cache.py.txt`.
- The included DSpark drafter was tested with a 14 GiB target GPU cache and
  84 GiB RAM, leaving room for the additional 6.88 GiB drafter weights. It ran
  at 5.08 tok/s versus 10.93/11.23 for surrounding warm baseline passes, accepted
  24 draft tokens and rejected 16, and first diverged at position 53. It remains
  disabled. `runtime.speculative_probe` is only a reproducible experiment.
- GPU-slot handle reuse was retained based on the full-suite comparison rather
  than the short warm diagnostic. It only applies when all experts in a layer
  have identical tensor layouts; mixed layouts retain the original allocation path.
  Its first measured source is in `experimental-slot-pool.py.txt`.
- The first `traffic/` attempt failed when formatting a disabled profiler; it is
  not a benchmark. `traffic-v2/` is the completed replacement.

Reproduction drivers are retained as `drivers/*.py.txt`. Historical cache/store
snapshots used by those drivers are in `variants/`; the original baseline is
also available from local commit `40d30bf`.
