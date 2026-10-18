"""Opt-in TP1 expert paging for the pinned DeepSeek V4.1 native graph.

Attention, Engram hashing/dequantization, routing and shared experts stay in
the native engine. Only routed-expert allocation and execution are replaced.
Validation and measured limits are recorded in README.md and results/.
"""
import re
import torch
from exllamav3.modules import BlockSparseMLP
from exllamav3.modules.module import Module


def cached_moe_class(cache):
    class CachedMoE(BlockSparseMLP):
        def __init__(self, *args, **kwargs):
            if args:
                raise ValueError("Adapter requires the pinned graph's keyword constructor")
            if kwargs.get("num_experts") != 384 or kwargs.get("router_type") != "sqrtsp" or kwargs.get("activation_fn") != "silu":
                raise ValueError("Adapter is restricted to the DeepSeek V4.1 Flash graph")
            match = re.fullmatch(r"layers\.(\d+)\.ffn", kwargs["key"])
            if not match:
                raise ValueError("Only main-model routed experts are supported")
            self.cache_layer = int(match.group(1))
            if any(f"{self.cache_layer}:{e}" not in cache.store.records for e in range(384)):
                raise ValueError("Cache does not own every routed expert")
            # The base class builds router/shared modules, but no routed Linear objects.
            kwargs["num_local_experts"] = 0
            super().__init__(**kwargs)
            self.expert_cache = cache

        def can_defer_load(self):
            return False

        def load(self, device, **kwargs):
            Module.load(self, device, **kwargs)
            if self.gates or self.ups or self.downs:
                raise RuntimeError("Routed expert allocation escaped the cache")
            for attr, key in (("e_score_correction_bias", self.e_score_correction_bias_key),
                              ("e_score_bias_vl", self.e_score_bias_vl_key)):
                if key:
                    value = self.config.stc.get_tensor(f"{self.key}.{key}", device,
                        optional=True, allow_bf16=True, no_defer=True)
                    setattr(self, attr, value.float() if value is not None else None)
            if self.e_score_correction_bias is None:
                raise ValueError("Missing native routing correction bias")
            self.load_routing(**kwargs)

        def forward(self, x, params, out_dtype=None):
            if params.get("activate_all_experts") or params.get("autosplit_measure"):
                raise ValueError("Only single-device inference is supported")
            y = x.reshape(-1, self.hidden_size).contiguous()
            ids, weights = self.routing_fn(y.shape[0], self.routing_cfg, y, params)
            stream = getattr(self.expert_cache, 'shared_stream', None)
            if (stream is not None and y.shape[0] == 1 and self.shared_experts is not None
                    and self.shared_experts.bc is not None):
                # The native shared-expert BC owns its projection scratch. The
                # grouped routed path also uses private scratch, so it cannot
                # overwrite the shared calculation's intermediates. Prefill
                # falls back to serial execution because its eager projections
                # use the engine's global single-row scratch buffers.
                current = torch.cuda.current_stream(x.device)
                stream.wait_stream(current)
                try:
                    with torch.cuda.stream(stream):
                        shared = self.shared_experts.forward(x, params)
                    result = self.expert_cache.apply(self.cache_layer, y, ids, weights,
                        limit=self.act_limit).view(x.shape)
                finally:
                    current.wait_stream(stream)
                    x.record_stream(stream)
                shared.record_stream(current)
                result = result + shared
            else:
                result = self.expert_cache.apply(self.cache_layer, y, ids, weights,
                    limit=self.act_limit).view(x.shape)
                if self.shared_experts is not None:
                    result = result + self.shared_experts.forward(x, params)
            return result.to(out_dtype) if out_dtype is not None else result

    return CachedMoE


def attach_engram(config, directory):
    """Add only the four original table tensors; keep quantized Engram projections."""
    from pathlib import Path
    from exllamav3.loader.safetensors import read_header
    stc = config.stc
    for layer, shard in ((1, 47), (14, 48)):
        filename = str(Path(directory) / f"model-{shard:05d}-of-00048.safetensors")
        header = read_header(filename, None)
        stc.file_headers[filename] = header
        stc.handles[filename] = None
        stc.tensor_files.append(filename)
        for suffix in ("weight", "scale"):
            name = f"layers.{layer}.engram.embed.{suffix}"
            if name in stc.tensor_file_map:
                raise ValueError(f"Unexpected duplicate Engram table: {name}")
            if name not in header:
                raise ValueError(f"Missing 1017 Engram table: {name}")
            stc.tensor_file_map[name] = filename
    stc.__dict__.pop("_get_tensor_file_map_trie", None)


def build_model(config, cache):
    from exllamav3 import Model
    from exllamav3.architecture import deepseek_v41
    original = deepseek_v41.BlockSparseMLP
    deepseek_v41.BlockSparseMLP = cached_moe_class(cache)
    try:
        model = Model.from_config(config)
    finally:
        deepseek_v41.BlockSparseMLP = original
    count = sum(isinstance(module, BlockSparseMLP) for module in model)
    if count != 40:
        raise ValueError(f"Expected 40 cached MoE blocks, found {count}")
    return model
