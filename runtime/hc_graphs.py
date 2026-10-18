"""Opt-in CUDA replay of the unchanged single-row hyperconnection mixer."""
import types
import torch


def configure(model, enabled):
    from exllamav3.modules.dsv41_hc import DSV41HyperConnection
    if not hasattr(model, '_recipe_mix_resources'):
        # All 80 sites execute serially, in model order. A shared capture stream
        # and pool reuse temporary allocations and the CUBLAS workspace. Each
        # site's persistent source/output tensors remain strongly referenced.
        model._recipe_mix_resources = {}
    count = 0
    for module in model:
        if not isinstance(module, DSV41HyperConnection):
            continue
        count += 1
        if not hasattr(module, '_recipe_mix_original'):
            module._recipe_mix_original = module.mix_full
            module._recipe_mix_graph = None
            module.mix_full = types.MethodType(_mix, module)
        module._recipe_mix_enabled = enabled
        module._recipe_mix_resources = model._recipe_mix_resources
    if count != 80:
        raise ValueError(f'Expected 80 DeepSeek mixers, got {count}')


def _mix(self, streams):
    if (not self._recipe_mix_enabled or streams.shape != (1, 1, 4, 5120)
            or streams.dtype != torch.float32 or not streams.is_contiguous()):
        return self._recipe_mix_original(streams)
    if self._recipe_mix_graph is None:
        # Warm the unchanged function before capture (CUBLAS and static native
        # workspaces initialize lazily). Mix reads its input; it never mutates it.
        source = torch.empty_like(streams)
        source.copy_(streams)
        current = torch.cuda.current_stream(streams.device)
        resources = self._recipe_mix_resources
        if not resources:
            resources.update(stream=torch.cuda.Stream(device=streams.device), pool=torch.cuda.graph_pool_handle())
        capture_stream = resources['stream']
        capture_stream.wait_stream(current)
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                self._recipe_mix_original(source)
        current.wait_stream(capture_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=resources['pool'], stream=capture_stream, capture_error_mode='thread_local'):
            outputs = self._recipe_mix_original(source)
        self._recipe_mix_source = source
        self._recipe_mix_outputs = outputs
        self._recipe_mix_graph = graph
    self._recipe_mix_source.copy_(streams)
    self._recipe_mix_graph.replay()
    return self._recipe_mix_outputs
