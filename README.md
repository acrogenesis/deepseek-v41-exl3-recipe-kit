# DeepSeek V4.1 Flash 1017 EXL3 on an RTX 5090

An experimental SSD-streaming recipe for one RTX 5090 (32 GB), a Ryzen 9950X3D,
128 GB RAM, and a Samsung 990 PRO NVMe. It is staged for this machine.
**Real-model generation is verified.** All 414 GB of model files are downloaded
and hash-checked. The recipe uses bounded CPU/GPU expert caches; starting GPU work
requires an explicit command.

The original [Sybil offload recipe](https://github.com/sybil-solutions/dsv41-flash-offload)
keeps every expert in pinned RAM. This recipe uses a bounded RAM cache and reads
misses from the original safetensors files. It does not create another 196 GB expert bank.

## Measured performance

| Configuration | Expert VRAM / RAM | Fresh tok/s | Repeat tok/s |
|---|---|---:|---:|
| Previous default | 22 / 64 GiB | 4.45 | 4.55 |
| Larger pageable cache | 22 / 84 GiB | 7.67 | 9.09 |
| Registered RAM, buffer and GPU-handle reuse (default) | 22 / 84 GiB | **9.51** | **12.31** |

The default improves this suite by **2.14x fresh / 2.71x repeat**.
An earlier run of the same configuration measured 8.98 / 12.34 tok/s. All full runs
passed **8/8 checks** and produced the **same 802 output tokens** across 14 requests.
The 30 tok/s target has not been reached. A short request fitting entirely in the
RAM cache reached about 19 tok/s in a separate instrumented diagnostic; that is
not the full-suite result above.

Same hardware and weights, no concurrent uploads/downloads, medians of three
128-token generations per phase, and an 8,192-token configured context. Prompts
contain 30/31/31 tokens; this does not fill the context window. Each request resets
KV/recurrent state; RAM and Engram page caches can warm. These are individual
sweeps, not confidence intervals. [Method](results/20261018-target30/METHOD.md),
[raw run](results/20261018-target30/selected-84g.json),
[summary](results/20261018-target30/summary.json).

The selected run peaked at **89.20 GiB** of container RAM and **27.16 GiB** of
PyTorch GPU allocations, with no process swap. Post-request samples left at least
**14.62 GiB** available host RAM and **2.21 GiB** free GPU memory. A 12 GiB
host reserve is enforced; other workloads may force cache eviction or stop a run.
Long contexts and other workloads still need validation.

## Model and memory

- [LibertAIDAI's calibrated 1017 EXL3 2.9 bpw checkpoint](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw),
  pinned to `c178a01284fc5c0e859d8668cffaac81ca8f4042`.
- The **1017** Engram tables from [DeepSeek's original release](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017),
  pinned to `1bae0e1088993b9b82de08f59de5be5a34e42170`, shards 47 and 48.
- About **414 GB** of downloads. SHA-256 verification for weights; Git blob verification for small metadata.
- Default cache limits: **84 GiB registered RAM + 22 GiB VRAM**, allocated on demand within a 96 GiB no-swap container.
- Initial benchmark: **8,192-token context**, one sequence, 256-token prefill chunks,
  greedy sampling, no drafter, text only. This is a validation profile, not a measured maximum context.

All 384 experts in all 40 layers remain available. Actual header inspection found
40,320 K3 and 5,760 K2 routed projection tensors. Stored main-model weights outside
the routed experts occupy about 6 GiB. Runtime memory measurements are below.
See [the inventory](results/expert-inventory.json) and [layout](results/checkpoint-layout.json).

## Preparation (CPU only)

```sh
python scripts/sources.py
bash scripts/build.sh
bash scripts/stage.sh
python scripts/headers.py
bash scripts/cpu-check.sh
python scripts/preflight.py
```

The build uses the existing local `localhost/glm53-sm120:20261001` image, whose ID is
locked in `runtime.lock.json`. It compiles the pinned ExLlama fork for **SM120** with
no GPU devices exposed. This base image must already exist on the target machine;
the repository is not yet a portable installation kit.

`stage.sh` starts a persistent user service. Defaults use this machine's Hugging Face
Python at `/home/archogenesis/.hf-cli/venv/bin/python`; set `HF_PYTHON` elsewhere.
Downloads resume in `work/models/`, with no second Hugging Face weight cache.
The service finishes with CPU checks, including every routed expert's MUL1 marker.
It never starts inference. Status:

```sh
journalctl --user -u deepseek41-exl3-download.service -f
python scripts/preflight.py
```

## When the GPU is free

These commands explicitly release the GPU hold for that invocation:

```sh
bash scripts/run.sh --gpu-is-free smoke
bash scripts/run.sh --gpu-is-free bench
```

The measured default is 22 GiB of expert VRAM. Use `--gpu-cache-gib 16` to leave
more GPU memory free. Use `--ram-cache-gib 64` for a smaller RAM budget,
and `--no-pinned-ram` to use pageable RAM with staged transfers.
`python scripts/campaign.py --allow-gpu` waits for verified downloads, checks a smoke
response at 16 and 22 GiB, then benchmarks both. If 22 GiB runs out of memory during
the smoke test, it tries 20 GiB. Results include a recommendation; defaults are not
changed automatically. The campaign waits for at least 30,000 MiB free GPU memory
before each run and stops if `work/GPU_HOLD` appears.

The first command loads the model and checks a short extraction answer. The second
runs the same eight functional checks and three fresh/repeated prompts as the GGUF
recipe, with identical prompt token IDs and context size. Every request gets fresh KV
state; repeated prompts can warm the expert cache. Reports save raw tokens, timings,
cache counters, runtime pins, and exact-check results under `results/`.

The EXL3 timer measures delivered tokens after the first token. Compare it with the
GGUF reports' **`client_decode_tok_s`**, not their internal server timer. Short capped
prompts and eight checks do not establish broad model quality. The validation below
compares full-vocabulary scores at fixed continuation tokens.
It measures the adapter against the same EXL3 checkpoint, not quantization loss
against unquantized DeepSeek.

## What changed

The native DeepSeek V4.1 graph comes from the pinned
[`vcruz305/exllamav3` fork](https://github.com/vcruz305/exllamav3).
Its attention, routing, shared experts, Engram hashing, and FP8 row dequantization
remain in place. Engram rows are read on the CPU and copied to the GPU; ATS is disabled
for this discrete GPU. Only the four table tensors are attached from the original
checkpoint, keeping the EXL3 checkpoint's Engram projection weights.

`runtime/native_adapter.py` replaces routed-expert allocation and execution with
the bounded cache. `runtime/indexed_store.py` reads byte ranges with Linux `O_DIRECT`
and retains hot experts in a capped RAM LRU. CUDA cache entries use stream-owning
leases and synchronize before eviction. Missing experts fail instead of being skipped.
The memory guard checks this process's cgroup; it does not require clearing desktop swap.

Validation after optimization:

- 15 CPU tests: byte-exact direct/buffered reads, split shards, gaps, eviction,
  cancellation, truncated files, marker validation and memory guards.
- 24 real-byte GPU cases: two CUDA streams, delayed DMA, RAM and GPU eviction,
  registered-buffer reuse, and failed-read cleanup.
- Eight resident-native versus paged MoE cases, K3 layer 0 and K2 layer 18:
  relative L2 error below 0.00077, cosine similarity above 0.9999996.
- Identical full-vocabulary scores at 64 teacher-forced positions across retained
  transfer paths, GPU-handle reuse, CPU thread counts and separate launches.
  Raw score arrays remain local; their hashes are in
  [the diagnostic evidence](results/20261018-target30/summary.json).

The initial launcher discarded ExLlama's kernel-tuning choices whenever its temporary
container exited. Separate launches could produce different tokens, including at
one fixed cache size. `run.sh` now persists cooperative-kernel and Triton caches in
`work/kernel-cache`. With saved tuning, the tested score arrays match byte for byte
across processes. This cache is local hardware/runtime state, not model weights.

The store combines adjacent source tensors into one O_DIRECT read per expert and
reuses its aligned read buffer. Registered RAM pages transfer directly to the GPU;
evicted equal-size buffers are recycled only after their DMA events complete.
GPU slots reuse quantized weight handles only for identical per-layer layouts.
Pageable RAM remains available with two bounded pinned staging buffers. Expert
selection and quantized tensor bytes are preserved.

The native grouped-expert loop and the included DSpark drafter were tested but
are disabled: neither met the speed/output criteria for this configuration.
Details and reproducible experiments are in the measurement report.

For a teacher-forced diagnostic and CPU profile:

```sh
bash scripts/run.sh --gpu-is-free diagnose \
  --output /recipe/results/my-diagnostic
```

This defaults to 16/22/16 GiB caches and the same 20 continuation tokens. Diagnostic
profiles include score copies and profiling overhead; they are excluded from speed
results. Normal `bench` reports use the unchanged streamed-token timer.

The earlier campaign in `results/20261017T232708Z-cache-campaign/` overlapped a GGUF
upload and did not persist tuning. Its measurements are retained for history; the
clean comparison is in `results/20261018-clean-optimization/`.

## Credits and license

Recipe code: AGPL-3.0. The GPU cache and geometry validator derive from
[`vllm-exl3`'s experimental NVMe components](https://github.com/vcruz305/vllm-exl3/tree/2a740a3773601c994072295fde45738ef8cea55e/experiments/dsv41_nvme).
Pristine copies, copyright notices, and licenses are retained in `runtime/vendor/`.
`runtime/cache.py` adds the desktop memory guard, exact expert ownership, pinned
transfers and single-token dispatch. ExLlamaV3 retains its MIT license. Model weights retain their own licenses.

The existing GGUF quant, reference logits, calibration, benchmark evidence, public Q2
comparison, and unrelated models were preserved. The 508 GB converted MXFP4/Q8
reference was deleted with the owner's approval and can be downloaded again through
the separate GGUF recipe.
