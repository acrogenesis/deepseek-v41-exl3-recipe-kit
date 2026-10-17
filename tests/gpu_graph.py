"""Diagnostic graph execution with zero Engram rows, never a model benchmark.

This finds GPU/runtime integration errors while the real table download runs.
Outputs cannot be used to measure the real model's quality or throughput.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser()
parser.add_argument("--allow-gpu", action="store_true")
parser.add_argument("--gpu-cache-gib", type=int, default=22)
args = parser.parse_args()
if not args.allow_gpu:
    raise SystemExit("Explicit --allow-gpu required")
from scripts.preflight import check
pending = [x for x in check()["pending"] if not (x.startswith("Pending: engram/") or x == "CPU staging validation is pending")]
if pending:
    raise SystemExit(str(pending))

import torch
from exllamav3 import Config, Tokenizer
from exllamav3.cache.cache import Cache
from exllamav3.generator import Generator, Job
from exllamav3.generator.sampler import GreedySampler
from exllamav3.modules.engram import EngramTable
from runtime.indexed_store import inventory, IndexedExpertStore, read_header
from runtime.cache import ExpertCache
from runtime.native_adapter import attach_engram, build_model

records, _ = inventory(ROOT)
store = IndexedExpertStore(ROOT / "work/models/exl3", records, 32 * 2**30)
capacity = args.gpu_cache_gib * 2**30
per = capacity // (40 * 4096) * 4096
quotas = {i: per for i in range(40)}
quotas[39] += capacity - sum(quotas.values())
cache = ExpertCache(store, capacity, layer_quotas=quotas)
result = {"status": "running", "synthetic_engram": True, "gpu_cache_gib": args.gpu_cache_gib,
    "scope": "Graph diagnostic only: zero Engram rows; no quality or speed measurement"}
output = ROOT / "results/gpu-graph-diagnostic.json"
model = None
try:
    config = Config.from_directory(str(ROOT / "work/models/exl3"))
    config.infer_params.no_reconstruct = True
    def engram_header(filename, fixes):
        base, header = read_header(ROOT / "work/headers/engram" / (Path(filename).name + ".header"))
        return {**header, "_header_offset": base}
    with patch("exllamav3.loader.safetensors.read_header", engram_header):
        attach_engram(config, ROOT / "work/models/engram")
    model = build_model(config, cache)
    kv = Cache(model, max_num_tokens=8192, max_batch_size=1)
    model.load("cuda:0", progressbar=False)
    tokenizer = Tokenizer.from_config(config)
    def zero_rows(self, ids, device):
        return torch.zeros((*ids.shape, self.head_dim), dtype=torch.float32, device=device)
    case = json.loads((ROOT / "fixtures/paired-gguf.json").read_text())["speed"][0]
    with patch.object(EngramTable, "rows", zero_rows), torch.inference_mode():
        generator = Generator(model=model, cache=kv, tokenizer=tokenizer, max_batch_size=1,
            max_chunk_size=256, recurrent_cache_size=512 * 2**20)
        generator.enqueue(Job(input_ids=torch.tensor([case["input_ids"]], dtype=torch.long),
            max_new_tokens=3, stop_conditions=[], sampler=GreedySampler()))
        count = 0
        while generator.num_remaining_jobs():
            for event in generator.iterate():
                if event.get("stage") == "error":
                    raise RuntimeError(str(event))
                if event.get("stage") == "streaming" and event.get("token_ids") is not None:
                    count += event["token_ids"].numel()
                    print(f"Graph diagnostic emitted {count} tokens (synthetic Engram)", flush=True)
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    if count != 3:
        raise RuntimeError(f"Expected 3 diagnostic tokens, got {count}")
    result.update(status="passed", generated_tokens=count, torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(), device_free_bytes=free,
        gpu_cache=dict(cache.stats), ram_cache=dict(store.stats))
except BaseException as exc:
    result.update(status="failed", error=repr(exc))
    raise
finally:
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if model is not None:
        model.unload()
    cache.close()
    store.close()
