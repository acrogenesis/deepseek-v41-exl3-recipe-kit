"""CPU-only construction against all real headers; never loads tensor payloads."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from exllamav3 import Config, Tokenizer
from exllamav3.loader.safetensors import SafetensorsCollection
from exllamav3.modules import BlockSparseMLP
from runtime.indexed_store import inventory, read_header
from runtime.native_adapter import build_model, attach_engram

ROOT = Path(__file__).resolve().parents[1]
assert not list(Path("/dev").glob("nvidia*")), "CPU contract test must not expose NVIDIA devices"
lock = json.loads((ROOT / "models.lock.json").read_text())
exl3 = next(m for m in lock["models"] if m["id"] == "exl3")


def metadata_files(self, directory, warn_if_override=True):
    for spec in exl3["files"]:
        if not spec["name"].endswith(".safetensors"):
            continue
        path = str(ROOT / "work/models/exl3" / spec["name"])
        base, header = read_header(ROOT / "work/headers/exl3" / (spec["name"] + ".header"), spec["bytes"])
        self.handles[path] = None
        self.tensor_files.append(path)
        self.file_headers[path] = {**header, "_header_offset": base}
        for name in header:
            if name != "__metadata__":
                self.tensor_file_map[name] = path


with patch.object(SafetensorsCollection, "add_tensor_files", metadata_files), \
     patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA initialization is forbidden")):
    config = Config.from_directory(str(ROOT / "work/models/exl3"))
    def engram_header(filename, fixes):
        base, header = read_header(ROOT / "work/headers/engram" / (Path(filename).name + ".header"))
        return {**header, "_header_offset": base}
    original_map = dict(config.stc.tensor_file_map)
    with patch("exllamav3.loader.safetensors.read_header", engram_header):
        attach_engram(config, ROOT / "work/models/engram")
    assert all(config.stc.tensor_file_map[k] == v for k, v in original_map.items())
    assert len(config.stc.tensor_file_map) - len(original_map) == 4
    records, report = inventory(ROOT, cached_headers=True)
    cache = SimpleNamespace(store=SimpleNamespace(records=records))
    model = build_model(config, cache)
    blocks = [m for m in model if isinstance(m, BlockSparseMLP)]
    assert len(blocks) == 40
    assert all(not (m.gates or m.ups or m.downs) for m in blocks)
    assert all(m.routing_gate is not None and m.shared_experts is not None for m in blocks)
    assert sum(".ffn.experts." in m.key for m in model) == 0
    from exllamav3.cache.cache import Cache
    kv = Cache(model, max_num_tokens=8192, max_batch_size=1)
    tokenizer = Tokenizer.from_config(config)
    fixture = json.loads((ROOT / "fixtures/paired-gguf.json").read_text())
    for case in fixture["checks"] + fixture["speed"]:
        actual = tokenizer.encode(case["rendered_prompt"], add_bos=False, encode_special_tokens=True).flatten().tolist()
        assert actual == case["input_ids"], f"Tokenization mismatch: {case['id']}"
    print(json.dumps({"gpu_devices_exposed": False, "cuda_initialized": torch.cuda.is_initialized(),
        "native_model_constructed": True, "moe_blocks": len(blocks), "resident_routed_linears": 0,
        "paired_prompt_tokenization_passed": len(fixture["checks"]) + len(fixture["speed"]),
        "engram_tables_attached_without_overrides": 4, "cache_constructed_context": 8192,
        "note": "Metadata construction only; forward/weight loading not tested"}, indent=2))
