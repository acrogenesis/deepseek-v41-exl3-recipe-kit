# Paging and dispatch measurements — 2026-10-18 UTC

The active goal is above 30 single-stream output tokens/s on the same DeepSeek
V4.1 Flash 1017 EXL3 2.9 bpw model. It has not been achieved. No checkpoint,
quantized bytes, expert selection, context configuration or sampling changed.
Runtime and model pins match the previous campaign. Defaults remain unchanged.

Hardware: RTX 5090 32 GB, Ryzen 9950X3D, 128 GB DDR5, Samsung 990 PRO on encrypted
Btrfs. Container RAM is capped at 96 GiB with zero process swap; the host reserve
is 12 GiB. RAM expert cache is 84 GiB. Most runs use 22 GiB of expert VRAM; the
last two full runs use 24 GiB. Each run is a separate container, with persisted
CUDA/Triton tuning. No other model or external transfer ran concurrently.

## Full workload and diagnostics

The unchanged fixture has eight exact checks, three 128-output-token prompts,
then those three prompts repeated. All 14 requests reset KV and recurrent state.
The context limit is 8192; the 30/31/31-token speed prompts do not fill it.
Throughput excludes the first output token. Report fresh and repeat medians
separately. Repeats benefit from expert-cache history; no KV/prompt-cache reuse.
These are individual sweeps, not confidence intervals. Raw outputs and memory
samples are retained in each JSON; summary.json checks all 802 output tokens
against the previously validated baseline.

`probe.json` instruments nested CPU paging operations across this full workload.
Its throughput is diagnostic only. Request counter differences include prefill,
so dividing by output count is an amortized cost, not a pure decode measurement.
The three repeats spent 1.22–2.01 seconds in preadv and 0.91–1.23 seconds in host
memory guards per 128 outputs. The counters are nested and must not be summed.

`PersistentMemoryGuard` keeps proc/cgroup files open but rereads all current
values at every existing check. It assumes the process stays in its container's
cgroup. CPU tests verify changed RAM availability, changed swap limits, nonzero
swap and descriptor cleanup. It never disables the RAM reserve or swap checks.

## Cache experiments

The original cache retains duplicate CPU copies of GPU-resident experts.
`VictimExpertStore` preferentially evicts these duplicates from RAM and copies
GPU-only records back to RAM before reusing their GPU slots. Dedicated CUDA
events protect CPU reads, pinned-buffer reuse, and GPU overwrites. Both transfers
and computation remain asynchronous where dependencies permit. Expert bytes are
copied exactly; no reconstruction or new quantization occurs.

The first version copied roughly 0.8–1.2 GB back to RAM per output token. This
cost offset much of the SSD saving. It remains available without a GPU-only cap
but is not recommended based on these measurements.

The selective version (`--gpu-only-gib 20`) removes CPU copies of the hottest
GPU experts first and bounds GPU-only data to approximately 20 GiB. RAM retains
copies for less frequently used GPU experts, avoiding most copy-back traffic.
Online access counts use only already observed routes. Simulations predicted
nearly zero repeat-prompt SSD reads and about 74–102 MB D2H/output at 22 GiB GPU,
compared with 836–891 MB for the unrestricted LFU victim cache. Actual counters
are retained to check the prediction. `exclusive-simulation.json` and
`hot-victim-simulation.json` are traffic simulations, not speed measurements.

A simulated global GPU LFU pool with uniform slots increased traffic versus
layer-local quotas; it was not implemented. Results are in global-gpu-simulation.json.

## Additional dispatch changes

All hyperconnection CUDA graphs now share one capture stream and memory pool.
Their persistent input/output tensors remain referenced, and sites replay in
model order. The 20-position score test measured approximately 37 MiB additional
live GPU allocations, versus roughly 1 GiB with separate graph resources. This
made the 24 GiB expert-cache trial possible. That short-context configuration
left only about 144 MiB free in one full run: it is an experimental benchmark
configuration, not a validated long-context serving default.

`--fused-reduce` replaces the six ordered expert multiply/add sequences with one
kernel. Explicit round-to-nearest PTX multiply and add preserve separate rounding,
subnormals and expert order; no FMA contraction. The K2/K3 GEMV and activation
arithmetic are unchanged. All 102 real-weight unit cases pass bit for bit,
including slot eviction/reuse. The unit microbenchmark excludes weight paging.

Three 20-position teacher-forced comparisons use only 16 GiB RAM to force eviction
and D2H traffic. Every vocabulary score matches the baseline. The final score
comparison covers the complete retained combination, including fused reduction.
NPY score arrays remain local and ignored; summary.json records their hashes.

## Reproduction

```sh
bash scripts/run.sh --gpu-is-free bench --gpu-cache-gib 24 \
  --grouped-dispatch --hc-graphs --gpu-policy lfu --victim-cache \
  --gpu-only-gib 20 --fast-guard --fused-reduce \
  --output /recipe/results/selective-victim.json
```

Use a new output filename. Keep tuning caches unchanged for score comparison.
The 22 GiB trial uses more GPU headroom but was slower on repeat prompts.
Exact drivers and rejected variants are retained alongside results. A budget
validation added after the fused benchmark does not change execution for that
valid configuration; variants/fused-bench-cache.py.txt matches its recorded hash.
