"""Explicit quiescent expert-cache resize between prefill and generation.

Writes back GPU-only bytes before replacing the arena. Optional preservation
restores selected resident weights without learning from those migration reads.
"""
import gc
import time

import torch

from .cache import CacheBusy
from .vendor.expert_store import ALIGN, Regions


def resize(cache, capacity_bytes, *, preserve=False):
    with cache.lock:
        if cache.closed:
            raise RuntimeError('Cache is closed')
        if cache.arena is not None and cache.arena.numel() == capacity_bytes:
            return None
    layers = sorted(cache.regions)
    if capacity_bytes <= 0 or capacity_bytes % ALIGN:
        raise ValueError('Capacity must be positive and aligned')
    per = capacity_bytes // (len(layers) * ALIGN) * ALIGN
    quotas = {layer: per for layer in layers}
    quotas[layers[-1]] += capacity_bytes - sum(quotas.values())
    for layer, size in quotas.items():
        largest = max(r['bytes'] for k, r in cache.store.records.items()
                      if int(k.split(':')[0]) == layer)
        if size < largest * (6 if cache.native_dispatch is not None else 1):
            raise ValueError('Resized quota cannot hold the required selected experts')
    gpu_only = getattr(cache.store, 'gpu_only_limit', None)
    if gpu_only is not None and capacity_bytes > cache.store.capacity + gpu_only:
        raise ValueError('RAM and GPU-only budgets cannot back the resized cache')
    with torch.cuda.device(cache.device), cache.lock:
        if cache.closed:
            raise RuntimeError('Cache is closed')
        if any(entry.users for entry in cache.entries.values()):
            raise CacheBusy('Cannot resize while a consumer holds a lease')
        before = cache.arena.numel()
        if before == capacity_bytes:
            return None
        started = time.perf_counter()
        resident_before = cache.resident_bytes
        retained = []
        if preserve:
            # Decisions use only observed frequency and existing residency.
            # No route prediction, future prompt inspection or quantization.
            order = {key: index for index, key in enumerate(cache.entries)}
            for layer in layers:
                used = 0
                keys = sorted(cache.layer_lru[layer],
                    key=lambda key: (cache.frequencies.get(key, 0), order[key]), reverse=True)
                for key in keys:
                    size = cache.entries[key].size
                    if used + size <= quotas[layer]:
                        retained.append(key)
                        used += size
            # Reinsert in original recency order after migrating the arena.
            retained.sort(key=order.__getitem__)
        frequencies = {key: cache.frequencies.get(key, 0) for key in retained}
        store_frequencies = {key: getattr(cache.store, 'frequencies', {}).get(key, 0) for key in retained}
        if cache.shared_stream is not None:
            cache.shared_stream.synchronize()
        # clear() waits for each entry's readers and writes GPU-only records back.
        cache.clear()
        cache.copy_stream.synchronize()
        if cache.writeback_stream is not None:
            cache.writeback_stream.synchronize()
        torch.cuda.current_stream(cache.device).synchronize()
        cache.arena = None
        gc.collect()
        torch.cuda.empty_cache()
        # Allocation failure is fatal to this run; close() remains safe. Never
        # keep stale projections pointing into the released allocation.
        cache.arena = torch.empty(capacity_bytes, dtype=torch.uint8, device=cache.device)
        base = 0
        regions = {}
        for layer in layers:
            regions[layer] = Regions(quotas[layer], base)
            base += quotas[layer]
        cache.regions = regions
        cache.stats['resizes'] = cache.stats.get('resizes', 0) + 1
        assert not cache.entries and not cache.slot_projections and cache.resident_bytes == 0
        disk_before = cache.store.stats['read_bytes']
        if preserve:
            try:
                for key in retained:
                    with cache.lease(key):
                        pass
            finally:
                # Restoring already resident weights is not a new model access.
                cache.frequencies.update(frequencies)
                if hasattr(cache.store, 'frequencies'):
                    cache.store.frequencies.update(store_frequencies)
                if getattr(cache.store, 'gpu_only_limit', None) is not None:
                    import heapq
                    cache.store.hot_duplicates = [(-cache.store.frequencies.get(key, 0), key)
                        for key in cache.store.duplicates]
                    heapq.heapify(cache.store.hot_duplicates)
            cache.copy_stream.synchronize()
            assert set(cache.entries) == set(retained)
        return dict(from_bytes=before, to_bytes=capacity_bytes,
                    evicted_gpu_bytes=resident_before, restored_experts=len(retained),
                    restored_gpu_bytes=cache.resident_bytes, restore_disk_bytes=cache.store.stats['read_bytes'] - disk_before,
                    seconds=time.perf_counter() - started)
