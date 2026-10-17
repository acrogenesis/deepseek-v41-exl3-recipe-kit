# DeepSeek V4.1 Flash 1017 EXL3 on an RTX 5090

An experimental SSD-streaming recipe for one RTX 5090 (32 GB), a Ryzen 9950X3D,
128 GB RAM, and a Samsung 990 PRO NVMe. It is staged for this machine.
**GPU inference and throughput have not been validated.** Nothing starts automatically.

The original [Sybil offload recipe](https://github.com/sybil-solutions/dsv41-flash-offload)
keeps every expert in pinned RAM. This recipe uses a bounded RAM cache and reads
misses from the original safetensors files. It does not create another 196 GB expert bank.

## Model and memory

- [LibertAIDAI's calibrated 1017 EXL3 2.9 bpw checkpoint](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw),
  pinned to `c178a01284fc5c0e859d8668cffaac81ca8f4042`.
- The **1017** Engram tables from [DeepSeek's original release](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017),
  pinned to `1bae0e1088993b9b82de08f59de5be5a34e42170`, shards 47 and 48.
- About **414 GB** of downloads. SHA-256 verification for weights; Git blob verification for small metadata.
- Proposed cache limits: **64 GiB RAM + 16 GiB VRAM**, allocated on demand within a 96 GiB no-swap container.
- Initial benchmark: **8,192-token context**, one sequence, 128-token prefill chunks,
  greedy sampling, no drafter, text only. This is a validation profile, not a measured maximum context.

All 384 experts in all 40 layers remain available. Actual header inspection found
40,320 K3 and 5,760 K2 routed projection tensors. Stored main-model weights outside
the routed experts occupy about 6 GiB; this is **not** a measured runtime footprint.
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

The first command loads the model and checks a short extraction answer. The second
runs the same eight functional checks and three fresh/repeated prompts as the GGUF
recipe, with identical prompt token IDs and context size. Every request gets fresh KV
state; repeated prompts can warm the expert cache. Reports save raw tokens, timings,
cache counters, runtime pins, and exact-check results under `results/`.

The EXL3 timer measures delivered tokens after the first token. Compare it with the
GGUF reports' **`client_decode_tok_s`**, not their internal server timer. Short capped
prompts and eight checks do not establish broad model quality. Full-model numerical
comparison and longer workloads remain part of GPU qualification.

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

Current validation: successful CPU compilation, 11 storage and memory-budget tests, byte-for-byte
reads of two real experts, and CPU construction of the full native model plus its 8K
cache. All 11 prompts tokenize identically to the paired GGUF fixture. See `results/`.
These checks do not prove that GPU kernels or full-model serving work together.

The initial cache uses demand reads and sequential expert dispatch. Its performance
on this machine is unknown; prefetch or fused dispatch should follow measurements.

## Credits and license

Recipe code: AGPL-3.0. The GPU cache and geometry validator derive from
[`vllm-exl3`'s experimental NVMe components](https://github.com/vcruz305/vllm-exl3/tree/2a740a3773601c994072295fde45738ef8cea55e/experiments/dsv41_nvme).
Pristine copies, copyright notices, and licenses are retained in `runtime/vendor/`.
`runtime/cache.py` changes their imports, desktop memory guard, and missing-expert
behavior. ExLlamaV3 retains its MIT license. Model weights retain their own licenses.

The existing GGUF quant, reference logits, calibration, benchmark evidence, public Q2
comparison, and unrelated models were preserved. The 508 GB converted MXFP4/Q8
reference was deleted with the owner's approval and can be downloaded again through
the separate GGUF recipe.
