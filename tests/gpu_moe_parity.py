"""Compare paged experts with the pinned engine's resident MoE implementation."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser()
parser.add_argument("--allow-gpu", action="store_true")
parser.add_argument('--output', type=Path, default=ROOT / 'results/gpu-moe-parity.json')
parser.add_argument('--pinned-ram', action='store_true')
args = parser.parse_args()
if not args.allow_gpu:
    raise SystemExit("Explicit --allow-gpu required")

import torch
from exllamav3 import Config, Model
from runtime.cache import ExpertCache
from runtime.indexed_store import inventory, IndexedExpertStore
from runtime.native_adapter import cached_moe_class

records, _ = inventory(ROOT, cached_headers=True)
store = IndexedExpertStore(ROOT / "work/models/exl3", records, 128 * 2**20, pinned=args.pinned_ram)
capacity = 4 * 2**30
per = capacity // (40 * 4096) * 4096
quotas = {i: per for i in range(40)}
quotas[39] += capacity - sum(quotas.values())
cache = ExpertCache(store, capacity, layer_quotas=quotas)
config = Config.from_directory(str(ROOT / "work/models/exl3"))
config.infer_params.no_reconstruct = True
model = Model.from_config(config)
receipt = json.loads((ROOT / "work/download-state.json").read_text())["verified"]
result = {"status": "running", "gpu": torch.cuda.get_device_name(), "cases": [],
          "pinned_ram": args.pinned_ram,
          "comparison": "Same checkpoint, resident native MoE versus paged MoE; identical router and shared experts",
          "relative_l2_limit": 0.002, "cosine_minimum": 0.99999}
output = args.output
output.resolve().relative_to((ROOT / 'results').resolve())
if output.exists():
    raise FileExistsError(output)


def save():
    output.write_text(json.dumps(result, indent=2) + "\n")


native = None
try:
    for layer in (0, 18):
        prefix = f"layers.{layer}.ffn"
        native = next(m for m in model if m.key == prefix)
        sources = {f for k, f in config.stc.tensor_file_map.items() if k.startswith(prefix + ".")}
        for name in sources:
            p = Path(name)
            st = p.stat()
            assert receipt["exl3/" + p.name]["identity"] == [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns]
        start = time.perf_counter()
        native.load(torch.device("cuda:0"))
        torch.cuda.synchronize()
        print(f"Loaded resident layer {layer} in {time.perf_counter()-start:.2f}s", flush=True)
        paged = cached_moe_class(cache)(config=config, key=prefix, hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size, num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok, key_up="experts.{expert_idx}.w3",
            key_gate="experts.{expert_idx}.w1", key_down="experts.{expert_idx}.w2",
            routing_gate=native.routing_gate, shared_experts=native.shared_experts,
            router_type="sqrtsp", activation_fn="silu", act_limit=config.swiglu_limit,
            routed_scaling_factor=config.routed_scaling_factor, interm_dtype=torch.half, out_dtype=torch.float)
        paged.routing_cfg = native.routing_cfg
        paged.device = native.device
        for rows in (1, 1, 8, 1):
            torch.manual_seed(1017 + rows)
            x = torch.randn((1, rows, config.hidden_size), dtype=torch.half, device="cuda:0")
            with torch.inference_mode():
                expected = native.forward(x, {}).float().clone()
                actual = paged.forward(x, {}).float()
                torch.cuda.synchronize()
                error = actual - expected
                relative = (error.norm() / expected.norm().clamp_min(1e-20)).item()
                cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
                passed = bool(torch.isfinite(actual).all()) and relative <= 0.002 and cosine >= 0.99999
                row = {"layer": layer, "rows": rows, "relative_l2": relative, "cosine": cosine,
                    "max_absolute_error": error.abs().max().item(), "passed": passed,
                    "gpu_cache": dict(cache.stats), "ram_cache": dict(store.stats)}
                result["cases"].append(row)
                save()
                print(json.dumps(row), flush=True)
                if not passed:
                    raise RuntimeError("Paged MoE differs from resident reference beyond FP16 tolerance")
        native.unload()
        native = None
        cache.clear()
        torch.cuda.empty_cache()
    result["status"] = "passed"
    result["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
except BaseException as exc:
    result.update(status="failed", error=repr(exc))
    raise
finally:
    save()
    if native is not None:
        native.unload()
    cache.close()
    store.close()
