"""Measure GPU allocation before Engram download completes; never run forward."""
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
parser.add_argument("--gpu-cache-gib", type=int, choices=range(1, 25), default=16)
args = parser.parse_args()
if not args.allow_gpu:
    raise SystemExit("Explicit --allow-gpu required")
from scripts.preflight import check
pending = [x for x in check()["pending"] if not (x.startswith("Pending: engram/") or x == "CPU staging validation is pending")]
if pending:
    raise SystemExit(str(pending))

import torch
from exllamav3 import Config
from exllamav3.cache.cache import Cache
from runtime.indexed_store import inventory, IndexedExpertStore, read_header
from runtime.cache import ExpertCache
from runtime.native_adapter import attach_engram, build_model

records, _ = inventory(ROOT)
store = IndexedExpertStore(ROOT / "work/models/exl3", records, 128 * 2**20)
capacity = args.gpu_cache_gib * 2**30
per = capacity // (40 * 4096) * 4096
quotas = {i: per for i in range(40)}
quotas[39] += capacity - sum(quotas.values())
cache = ExpertCache(store, capacity, layer_quotas=quotas)
result = {"status": "running", "gpu_cache_gib": args.gpu_cache_gib, "context_tokens": 8192,
    "scope": "Actual GPU load of all main-model weights and KV state; Engram table headers only; no forward pass"}
output = ROOT / f"results/gpu-load-{args.gpu_cache_gib}g.json"
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
    started = time.perf_counter()
    model.load("cuda:0", progressbar=False)
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    result.update(status="passed", load_seconds=time.perf_counter()-started,
        torch_allocated_bytes=torch.cuda.memory_allocated(), torch_reserved_bytes=torch.cuda.memory_reserved(),
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(), device_free_bytes=free, device_total_bytes=total)
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
