"""Demand-only packed EXL3 cache with explicit CUDA lifetimes.

Derived from vcruz305/vllm-exl3 experiments/dsv41_nvme, pinned in runtime.lock.json.
Local changes: desktop cgroup guard, fail-closed TP1 ownership, pinned transfer
buffers, layer-local LRU lookup, and single-row dispatch without GPU nonzero.
Experimental C1/eager path. It never allocates the full expert checkpoint.
"""
from collections import OrderedDict
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass, field
from pathlib import Path
import threading
import struct
import torch
from exllamav3.modules.quant.exl3 import LinearEXL3
from .vendor.expert_store import ALIGN, Regions
from .memory_budget import guard_memory


def host_memory():
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return {name: int(values[name].split()[0])*1024 for name in ('MemAvailable','SwapTotal','SwapFree')}


class CacheBusy(RuntimeError):
    pass


@dataclass
class Entry:
    key: str
    offset: int
    size: int
    generation: int
    projections: dict
    ready: object = None
    users: int = 0
    events: list = field(default_factory=list)


class ExpertCache:
    def __init__(self, store, capacity_bytes, *, layer_quotas=None, device='cuda:0', reserve_bytes=8*2**30, native_dispatch=False, grouped_dispatch=False, route_trace=False, policy='lru'):
        if capacity_bytes <= 0 or capacity_bytes % ALIGN:
            raise ValueError('Cache capacity must be aligned')
        self.store = store
        if policy not in ('lru', 'lfu'):
            raise ValueError('Unknown expert eviction policy')
        self.policy = policy
        self.frequencies = {}
        self.route_trace = [] if route_trace else None
        self.device = torch.device(device)
        self.reserve = reserve_bytes
        guard_memory(reserve_bytes)
        layers = sorted({int(key.split(':')[0]) for key in store.records})
        if layer_quotas is None:
            if len(layers) != 1:
                raise ValueError('Multiple layers require explicit byte quotas')
            layer_quotas = {layers[0]: capacity_bytes}
        if set(layer_quotas) != set(layers) or sum(layer_quotas.values()) != capacity_bytes:
            raise ValueError('Layer quotas must exactly partition the cache')
        self.regions = {}
        base = 0
        for layer, size in sorted(layer_quotas.items()):
            self.regions[layer] = Regions(size, base)
            largest = max(r['bytes'] for key,r in store.records.items() if int(key.split(':')[0]) == layer)
            if largest > size:
                raise ValueError('Layer quota cannot fit its largest expert')
            if (native_dispatch or grouped_dispatch) and 6 * largest > size:
                raise ValueError('Grouped dispatch needs room for six experts per layer')
            base += size
        self.arena = torch.empty(capacity_bytes, dtype=torch.uint8, device=self.device)
        self.entries = OrderedDict()
        self.layer_lru = {layer: OrderedDict() for layer in layers}
        self.lock = threading.RLock()
        self.generation = 0
        self.stats = dict(hits=0, misses=0, evictions=0, failures=0, peak_resident_bytes=0)
        self.resident_bytes = 0
        self.closed = False
        # Two bounded pinned buffers let disk/RAM staging overlap queued GPU
        # work. An event protects each buffer until its DMA read has finished.
        largest = max(r['bytes'] for r in store.records.values())
        self.staging = [torch.empty(largest, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
        self.staging_events = [None, None]
        self.staging_index = 0
        self.copy_stream = torch.cuda.Stream(device=self.device)
        # A cache slot's GPU addresses stay fixed when its expert changes.
        # Reuse the quantized views/BC handles where every expert in a layer
        # has identical geometry. No reconstructed weight is retained.
        layouts = {layer: set() for layer in layers}
        for key, record in store.records.items():
            layout = (record['bytes'], tuple((name, spec['offset'], spec['bytes'],
                spec['dtype'], tuple(spec['shape'])) for name, spec in sorted(record['tensors'].items())))
            layouts[int(key.split(':')[0])].add(layout)
        self.reusable_layers = {layer for layer, options in layouts.items() if len(options) == 1}
        self.slot_projections = {}
        self.native_dispatch = None
        self.grouped_dispatch = grouped_dispatch
        if native_dispatch or grouped_dispatch:
            from .native_dispatch import load_dispatch
            self.native_dispatch = load_dispatch()

    def _guard(self):
        guard_memory(self.reserve)

    def _evict(self, key):
        entry = self.entries[key]
        if entry.users:
            raise CacheBusy('Cannot evict an expert in use')
        for event in entry.events:
            event.synchronize()
        entry.events.clear()
        # Dropping handles is necessary: LinearEXL3.unload() is a no-op.
        entry.projections = {}
        self.regions[int(key.split(':')[0])].release(entry.offset)
        self.resident_bytes -= entry.size
        del self.entries[key]
        del self.layer_lru[int(key.split(':')[0])][key]
        self.stats['evictions'] += 1

    def _load(self, key, cancel):
        self._guard()
        record = self.store.records[key]
        layer = int(key.split(':')[0])
        region = self.regions[layer]
        offset = region.allocate(record['bytes'])
        while offset is None:
            candidates = (k for k in self.layer_lru[layer] if self.entries[k].users == 0)
            candidate = (next(candidates, None) if self.policy == 'lru'
                else min(candidates, key=self.frequencies.__getitem__, default=None))
            if candidate is None:
                raise CacheBusy('All eviction candidates are in use')
            self._evict(candidate)
            offset = region.allocate(record['bytes'])
        projections = {}
        try:
            with self.store.read(key, cancel=cancel) as data:
                for p in ('w1', 'w2', 'w3'):
                    if struct.unpack_from('<i', data, record['tensors'][f'{p}.mul1']['offset'])[0] != -2082680531:
                        raise ValueError('MUL1 marker mismatch')
                cpu = torch.frombuffer(data, dtype=torch.uint8)
                direct_dma = getattr(self.store, 'pinned', False)
                if direct_dma:
                    if not cpu.is_pinned():
                        raise RuntimeError('RAM cache registration was lost')
                    pinned = cpu
                else:
                    slot = self.staging_index
                    self.staging_index = (slot + 1) % len(self.staging)
                    if self.staging_events[slot] is not None:
                        self.staging_events[slot].synchronize()
                    pinned = self.staging[slot][:record['bytes']]
                    pinned.copy_(cpu)
                target = self.arena.narrow(0, offset, record['bytes'])
                # Eviction waits for all old consumers before this slot is
                # reused. A separate stream can copy the next expert while
                # the owning stream is still computing the previous one.
                try:
                    with torch.cuda.stream(self.copy_stream):
                        target.copy_(pinned, non_blocking=True)
                        ready = torch.cuda.Event()
                        ready.record(self.copy_stream)
                    if direct_dma:
                        self.store.protect_dma(key, ready)
                    else:
                        self.staging_events[slot] = ready
                except BaseException:
                    # Still holding the RAM lease: no other caller can unmap
                    # a registered source before an exceptional DMA completes.
                    self.copy_stream.synchronize()
                    raise
                finally:
                    del cpu, pinned
            if cancel is not None and cancel.is_set():
                raise InterruptedError('Expert load cancelled')
            slot_key = (layer, offset)
            projections = self.slot_projections.get(slot_key, {}) if layer in self.reusable_layers else {}
            if not projections:
                dtype = {'I16':torch.int16, 'F16':torch.float16, 'I32':torch.int32}
                for p in ('w1','w2','w3'):
                    tensors = {}
                    for suffix in ('trellis','suh','svh','mul1'):
                        spec = record['tensors'][f'{p}.{suffix}']
                        tensors[suffix] = target.narrow(0,spec['offset'],spec['bytes']).view(dtype[spec['dtype']]).view(spec['shape'])
                    inner = LinearEXL3(config=None, in_features=tensors['suh'].numel(),
                        out_features=tensors['svh'].numel(), out_dtype=torch.float16,
                        transformers_fix=True, **tensors)
                    inner.config.infer_params.no_reconstruct = True
                    projections[p] = inner
                if layer in self.reusable_layers:
                    self.slot_projections[slot_key] = projections
            self.generation += 1
            entry = Entry(key,offset,record['bytes'],self.generation,projections,ready)
            self.entries[key] = entry
            self.layer_lru[layer][key] = None
            self.resident_bytes += record['bytes']
            self.stats['peak_resident_bytes'] = max(self.stats['peak_resident_bytes'],self.resident_bytes)
            return entry
        except BaseException:
            self.copy_stream.synchronize()
            torch.cuda.current_stream(self.device).synchronize()
            projections = {}
            region.release(offset)
            self.stats['failures'] += 1
            raise

    @contextmanager
    def lease(self, key, *, cancel=None):
        """Lease one expert, bound to the CUDA stream current at acquisition.

        Stream contract: the owning stream is captured per LEASE in a local
        variable, never on the shared `Entry` (one entry can be leased by
        several callers at once, each on its own stream). The consumer must
        enqueue its work for this lease on the captured stream - normally by
        acquiring the lease inside `torch.cuda.stream(s)` - and must not switch
        streams inside the lease body. On release the ordering event is recorded
        on that same captured stream, so `_evict()`'s `event.synchronize()`
        covers the kernels the consumer actually enqueued even when the release
        happens after the caller's `torch.cuda.stream(s)` block has exited.
        """
        with torch.cuda.device(self.device), self.lock:
            if self.closed:
                raise RuntimeError('Cache is closed')
            if key not in self.store.records:
                raise KeyError(key)
            if cancel is not None and cancel.is_set():
                raise InterruptedError('Expert lease cancelled')
            if self.policy != 'lru':
                self.frequencies[key] = self.frequencies.get(key, 0) + 1
            entry = self.entries.get(key)
            if entry is None:
                self.stats['misses'] += 1
                entry = self._load(key,cancel)
            else:
                self.stats['hits'] += 1
                entry.events[:] = [e for e in entry.events if not e.query()]
                self.entries.move_to_end(key)
                self.layer_lru[int(key.split(':')[0])].move_to_end(key)
            entry.users += 1
            owning_stream = torch.cuda.current_stream(self.device)
            owning_stream.wait_event(entry.ready)
        try:
            yield entry
        finally:
            with torch.cuda.device(self.device), self.lock:
                event = torch.cuda.Event()
                event.record(owning_stream)
                entry.events.append(event)
                entry.users -= 1

    def apply(self, layer, x, ids, weights, *, limit=10.0):
        self._guard()
        if x.ndim != 2 or ids.ndim != 2 or ids.shape != weights.shape or ids.shape[0] != x.shape[0]:
            raise ValueError('Invalid routing shapes')
        if x.device != self.device or ids.device != self.device or weights.device != self.device:
            raise ValueError('Routing and activations must use the cache device')
        result = torch.zeros_like(x,dtype=torch.float32)
        # Decode routes one row. Copy its six IDs once and preserve the old
        # sorted expert order without six GPU nonzero/index_select operations.
        single = x.shape[0] == 1
        routes = ids[0].tolist() if single else None
        if single and len(set(routes)) != len(routes):
            raise ValueError('Duplicate experts in a routing row')
        ordered = sorted(enumerate(routes), key=lambda pair: pair[1]) if single else [(None, eid) for eid in torch.unique(ids).tolist()]
        if self.route_trace is not None:
            self.route_trace.append([layer, [int(eid) for _, eid in ordered]])
        if single and self.native_dispatch is not None:
            with ExitStack() as leases:
                projections = [leases.enter_context(self.lease(f'{layer}:{eid}')).projections for _, eid in ordered]
                dispatch = self.native_dispatch.grouped_experts if self.grouped_dispatch else self.native_dispatch.exact_experts
                return dispatch(x.half().contiguous(), weights,
                    [position for position, _ in ordered],
                    [p['w1'].bc for p in projections], [p['w3'].bc for p in projections],
                    [p['w2'].bc for p in projections], limit)
        for position, eid in ordered:
            key = f'{layer}:{int(eid)}'
            if key not in self.store.records:
                raise ValueError(f"Unowned routed expert: {key}")
            if not single:
                rows, positions = (ids == eid).nonzero(as_tuple=True)
            with self.lease(key) as entry:
                h = (x if single else x.index_select(0,rows)).half().contiguous()
                p = entry.projections
                gate = p['w1'].forward(h,{},out_dtype=torch.float32)
                up = p['w3'].forward(h,{},out_dtype=torch.float32)
                if limit is not None:
                    gate = gate.clamp(max=limit)
                    up = up.clamp(-limit,limit)
                act = (torch.nn.functional.silu(gate)*up).half().contiguous()
                down = p['w2'].forward(act,{},out_dtype=torch.float32)
                if single:
                    result.add_(down * weights[:, position:position + 1].float())
                else:
                    result.index_add_(0,rows,down*weights[rows,positions,None].float())
        return result

    def clear(self):
        with self.lock:
            if any(e.users for e in self.entries.values()):
                raise CacheBusy('Cache has active leases')
            for key in list(self.entries):
                self._evict(key)
            self.slot_projections.clear()

    def close(self):
        with self.lock:
            self.clear()
            for event in self.staging_events:
                if event is not None:
                    event.synchronize()
            self.staging.clear()
            self.staging_events.clear()
            self.arena = None
            self.closed = True
