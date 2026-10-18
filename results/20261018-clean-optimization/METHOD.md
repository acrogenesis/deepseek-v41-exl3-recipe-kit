# Comparison method

RTX 5090 32 GB, Ryzen 9950X3D, 128 GB DDR5, Samsung 990 PRO 4 TB NVMe;
Torch 2.13.0+cu130, CUDA 13.0, driver 610.57.04. One GPU/one sequence,
8,192-token context, 256-token prefill chunks, 64 GiB explicit RAM expert cache,
96 GiB container limit with no process swap, 12 GiB host reserve. Exact native
routing; the 1017 EXL3 checkpoint and original 1017 Engram tables in models.lock.json.

The eight checks and three 30/31/31-token speed prompts come from the unchanged
paired GGUF fixture. Each speed prompt generates 128 tokens greedily, once in the
fresh phase and once in the repeat phase. Each request resets KV/recurrent state.
“Fresh” means no reused prompt KV state, not a cold SSD, RAM expert cache or Engram
OS page cache. The latter may warm between processes. No upload or download ran
during these tests. No global OS cache flushing or swap configuration changes.

The metric is delivered output tokens after the first token divided by time from
the first to last delivered token. TTFT is recorded separately. Medians are over
three prompts per phase, not many independent sweeps. Raw token IDs and individual
stream timestamps are retained in each JSON report.

- `baseline-16g.json`: original code, with the original ephemeral tuning behavior.
- `baseline-fixed-16g.json`: original cache/store from commit a76fa3c, with saved tuning.
- `optimized-16g.json`, `optimized-22g.json`: optimized code, same saved tuning.

The final three runs execute sequentially in that order. The exact local drivers
are retained as `benchmark-driver.py.txt` and `baseline-driver.py.txt`; the baseline
cache/store files are recoverable with `git show a76fa3c:runtime/<name>.py`. Reports include hashes of
the loaded cache/store source and the tuning file before/after each run. The tuning
file snapshots are retained here. The first baseline is excluded from the strict
speed comparison because its kernel choices were not shared with the optimized runs.

## Correctness and profile diagnostics

`diagnostic-baseline`: original implementation, one model loaded once, 16/22/16 GiB
cache sequence. All 20 forced positions had identical full-vocabulary logits.

`diagnostic-optimized`: baseline cache followed by optimized 16 and 22 GiB caches,
using the coalesced disk reader throughout. Score arrays match byte for byte.

`diagnostic-restart`: separate process, optimized 16/22 GiB, same persisted kernel
tuning. Score arrays match all three arrays from `diagnostic-optimized` byte for byte.
This also includes the subsequent layer-local LRU lookup change.

The source metadata in each diagnostic captures that version. Diagnostic times
include cProfile, logits copied to the CPU and cold first-use compilation. They
are not presented as generation throughput. The raw NPY and pstats files remain
local and are omitted from Git; `logit-artifacts.json` records their score hashes.
The text profiles and JSON comparisons are committed.

The old temporary containers discarded native autotune choices. A fresh process
could change outputs even without changing cache size. Persisting kernel caches
eliminated that variation in the tested score comparisons; this is a workload-level
observation, not a proof of determinism for every context or backend operation.

GPU tests additionally cover sixteen real-byte transfer/lifetime cases across two
streams and eight resident-native versus paged MoE comparisons on K2/K3 experts.
They preserve source bytes and check execution; they do not measure the quantizer's
accuracy relative to an unquantized model.

## Default selected after the comparison

The 22 GiB cache matched all 802 baseline tokens and passed 8/8 checks. It was
5.71% faster than optimized 16 GiB on fresh medians and 5.62% faster on repeats,
with 27.16 GiB peak PyTorch allocation and at least 2.46 GiB device memory free
after requests. The local 8K profile now defaults to 22 GiB. The GPU cache is only
one part of total VRAM use; `--gpu-cache-gib 16` retains more headroom. The profile
and lock validation labels were updated after the measured runs completed.
