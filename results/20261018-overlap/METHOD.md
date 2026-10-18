# Expert-transfer overlap measurements — 2026-10-18 UTC

Two independent full sweeps exceeded **30 output tokens/s on every repeat
prompt**. Their repeat medians were **30.60 and 30.75 tok/s**; fresh medians were
**16.18 and 15.73 tok/s**. This achieves the 30+ target for the measured repeat
workload. Fresh prompts still incur substantial SSD reads.

Hardware, checkpoint, engine and image pins match the
[paging campaign](../20261018-paging/METHOD.md): RTX 5090, Ryzen 9950X3D,
128 GB RAM, EXL3 2.9 bpw and original 1017 Engram tables. Expert caches are
24 GiB GPU and 84 GiB pinned RAM. The container is capped at 96 GiB with zero
process swap and a 12 GiB host reserve. No system clock/power settings changed.

## Full workload

The unchanged fixture has eight exact checks, then three 128-output-token
prompts fresh and repeated, with fresh KV/recurrent state on every request.
Each run produced the same 802 baseline tokens across all 14 requests and
passed all eight checks. The context limit is 8192; speed prompts contain
30/31/31 tokens, not a filled context. Throughput excludes the first output
token. These are individual sweeps, not confidence intervals.

| Configuration | Fresh tok/s | Repeat tok/s | Result file |
|---|---:|---:|---|
| Compute resident experts before staging misses | 12.57 | 27.91 | full.json |
| Same + broader HC graphs | 12.83 | 25.73 | block-full.json |
| Start DMA before resident computation | 12.87 | 28.83 | dma-full.json |
| Same + shared-expert overlap | 14.34 | 29.41 | shared-full.json |
| Same + host thread pinned to CPU 6 | 12.88 | 23.06 | affinity-full.json |
| Shared overlap + online Engram row cache | **16.18** | **30.60** | engram-full.json |
| Independent repeat of that configuration | **15.73** | **30.75** | engram-replica.json |

The six final repeat rates were 30.5429 / 30.9590 / 30.5977 and
30.5950 / 30.9458 / 30.7545. Both runs used identical runtime source hashes and
unchanged tuning caches. Neither broader HC graphs nor CPU affinity is retained
as an active option. Sources and drivers for those trials remain archived.

## Retained scheduling changes

`--overlap-loads` leases selected resident experts before any miss can evict
them, starts missing-expert DMA, then computes the resident group before waiting
for the uploads. A final kernel combines both groups in the original sorted
expert order using separate round-to-nearest FP32 multiply/add instructions.
Deferred leases also wait on upload readiness during exceptional cleanup.

`--overlap-shared` runs the native shared expert on a separate stream while
routed experts are staged. Its buffers are separate from the grouped routed
kernel's scratch. Input/output stream dependencies protect tensor lifetimes;
shared sites still complete in model order. Prefill retains serial execution.
The final sum retains the original arithmetic order.

`--engram-cache-rows 65536` adds one bounded online LRU across both Engram layers.
It retains original weight and scale bytes; misses call the original reader.
The existing FP8/e8m0 dequantization receives the same bytes. No future prompts
or rows are preloaded. It avoids sending tiny repeated reads through the
engine's Python thread pools. The full fixture retained 59710 rows with
7881720 payload bytes (7.52 MiB, plus Python metadata), with no new row reads
during the three repeats. CPU tests separately exercise eviction and batches
larger than the cache capacity.

All options are disabled by default. No checkpoint bytes, expert selection,
activation precision, quantization, context configuration or sampling changed.

## Validation

- 19 CPU tests, including row ordering, duplicates, handle isolation, bounded
  eviction and failed-read cleanup.
- 252 real-weight K2/K3 cases: interleaved groups of every split size, both
  activation limits, multiple input scales, delayed uploads and slot reuse.
- 28 byte/lifetime cases: exceptional deferred leases, GPU-only writeback,
  cross-stream leases and registered-RAM eviction during DMA.
- Five 20-position comparisons of all 129280 vocabulary scores, exactly equal
  to baseline. A 16 GiB RAM cache forces real paging/writeback. The final
  comparison includes 128 actual Engram-cache hits. Local ignored NPY files
  have matching hashes recorded in summary.json.
- All 802 output tokens match in every complete full run, including both final
  sweeps. No process swap. Final runs peaked at 88.32 / 88.12 GiB container RAM;
  sampled host availability stayed above 14.27 GiB.

## Diagnostics and rejected paths

`profile.json` profiles eight decode steps of the first repeat in the
compute-first variant. It is excluded from speed comparisons. The trace records
about 3084 kernels/token, 15.25 ms kernel activity and 11.76 ms memcpy activity,
with no temporal overlap. CPU staging often delayed the first upload until
resident computation had finished. These figures include profiling effects and
are not predictions of another scheduler's latency.

The host probe adds GC callbacks and samples the dispatch CPU. GC cost was
about 0.01 seconds per 128-output repeat request. The thread mostly ran on
CPUs 8/9/24/25, on the 32 MiB L3 chiplet. Pinning it to CPU 6 on the 96 MiB
chiplet after initializing the 16-worker CPU pool did not help overall.

External NVML samples at 200 ms plus request wall-clock markers are also
excluded from speed comparisons. Repeat rates varied from 23.47 to 31.02 while
median SM clocks stayed at 2857 MHz and memory at 13801 MHz, with no power-cap
or thermal-slowdown samples. Those clocks do not explain the large variation.

## Reproduction

```sh
bash scripts/run.sh --gpu-is-free bench --gpu-cache-gib 24 \
  --grouped-dispatch --hc-graphs --gpu-policy lfu --victim-cache \
  --gpu-only-gib 20 --fast-guard --fused-reduce \
  --overlap-loads --overlap-shared --engram-cache-rows 65536 \
  --output /recipe/results/my-fast-bench.json
```

Use a new output filename. Exact drivers and earlier source variants are retained
under drivers/ and variants/. The final runs left about 154 MiB GPU memory free.
The 24 GiB cache remains opt-in; larger contexts and concurrent GPU applications
have not been validated with it.
