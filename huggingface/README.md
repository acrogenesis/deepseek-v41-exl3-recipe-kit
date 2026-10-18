---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- exl3
- exllamav3
- deepseek_v41
- moe
- quantized
- rtx-5090
---

# DeepSeek V4.1 Flash 1017 EXL3 on one RTX 5090

**30.68 tok/s on repeated prompts; 15.96 tok/s on fresh prompts** on an RTX 5090
32 GB, Ryzen 9950X3D, 128 GB RAM and Samsung 990 PRO NVMe.

![Generation speed](assets/original-speed.png)

This packages [LibertAIDAI's 2.9-bpw EXL3 quant](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)
with the required Engram tables from the exact
[DeepSeek 1017 release](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).
**The model weights are unchanged.** This is a mirror and deployment bundle,
not a new quantization or fine-tune. Our work is the bounded SSD/RAM/GPU streaming
recipe and the measurements on this PC.

[Recipe and source code on GitHub](https://github.com/acrogenesis/deepseek-v41-exl3-recipe-kit).
The recipe is experimental, Linux/Podman-based, and currently requires its pinned
local base image. Its README documents that prerequisite. This bundle is not
directly loadable with stock Transformers or upstream ExLlama.

## Measurements

One sequence, greedy sampling, thinking off, no drafter, 8,192 allocated context.
Every request resets KV/recurrent state; expert and Engram caches can warm.
Fresh prompts follow eight functional checks. No cold-disk claim.

| Workload | EXL3 fresh / repeated | Our IQ2_XXS GGUF fresh / repeated |
|---|---:|---:|
| Original, 128 outputs | 15.96 / 30.68 tok/s | 27.95 / 36.04 tok/s |
| Six additional topics, 128 outputs | 15.76 / 17.38 tok/s | 24.08 / 29.89 tok/s |
| Original, 512 outputs | 14.00 / 17.31 tok/s | 27.35 / 30.05 tok/s |
| 8,063 inputs, 128 outputs | 11.29 / 11.43 tok/s | 25.71 / 32.65 tok/s |

The original row pools two independent sweeps per model, six requests per phase.
EXL3 uses 24 GiB GPU / 84 GiB RAM there, and 23/84 for the expanded workloads.
GGUF uses 18/84. The 512-output GGUF row pools two sweeps; EXL3 uses one.
The six-topic panel uses one sweep per model; the long-input row has one request
per phase. Longer answers and different prompts can be substantially slower.

With equal 18 GiB GPU / 84 GiB RAM expert caches, EXL3 measured 14.85 fresh /
19.64 repeated tok/s on the original workload. Equal bytes cover different
fractions of these differently quantized expert sets.

## Measured quality

![Held-out quality](assets/quality.png)

| Same 2,040 held-out token positions | EXL3 | Our IQ2_XXS GGUF |
|---|---:|---:|
| Perplexity, lower is better | 6.405124 | 7.520919 |
| Mean KL to converted reference, lower is better | 0.130094 | 0.362934 |
| Reference top-token agreement | 85.637% | 77.549% |
| Separate exact-answer panel | 20/24 | 11/24 |

Eight 512-token WikiText-2 chunks, separate from calibration. Reference perplexity
is 6.335829. The reference retains native MXFP4 experts with Q8 dense/Engram
conversion; it is not an exact native forward pass. This compares complete
quantized model/runtime packages, not file formats in isolation or a percentage
of intelligence. Exact-answer checks also penalize extra text, formatting and
truncated answers. Both packages made arithmetic errors.

Raw [performance results](https://github.com/acrogenesis/deepseek-v41-exl3-recipe-kit/tree/main/results/20261018-overlap),
[expanded comparison](https://github.com/acrogenesis/deepseek-v41-exl3-recipe-kit/tree/main/results/20261018-gguf-comparison),
and [held-out scores](https://github.com/acrogenesis/deepseek-v41-exl3-recipe-kit/blob/main/results/20261018-gguf-comparison-quality/quality.json)
are retained in the recipe.

## Files and runtime

- Root: all 43 original EXL3 shards, config, tokenizer, index and K map.
- `engram/`: original 1017 shards 47 and 48 and their source metadata.
- About **413.91 GB** for the original files. All 384 experts in all 40 layers
  remain available. No expert pruning or routing bias.
- `source-manifest.json`: exact source revisions, filenames, sizes and hashes.
- `provenance/LibertAIDAI-README.md`: original quantization documentation.

The runtime uses bounded pinned RAM and GPU expert caches, grouped expert kernels,
hyperconnection graphs, exact-byte RAM writeback, overlapping expert transfers,
and a bounded Engram row cache. The tested profile is text-only; vision, tool
calling and DSpark are not enabled. The 8K allocation is not a maximum-context
claim. Use the pinned fork in the recipe.

To download the complete bundle:

```bash
hf download acrogenesis/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-5090 \
  --local-dir ./deepseek41-exl3-5090
```

The recipe's existing downloader fetches the same verified bytes from the two
original source repositories into `work/models/exl3` and `work/models/engram`.
There is no need to download both that layout and this mirror.

## Credits and license

DeepSeek created the model. LibertAIDAI created this EXL3 quantization.
The runtime builds on the pinned vcruz305 ExLlama and NVMe-cache work, with the
Sybil/MiaAI serving recipes providing upstream context. acrogenesis adapted and
measured the bounded 5090 recipe.

Model weights retain DeepSeek's **MIT license**, included in `LICENSE`.
The recipe source is separately licensed **AGPL-3.0** and includes its upstream
notices. Source revisions are pinned in the manifest and the GitHub lock files.
