"""Bounded RAM victim cache: discard GPU duplicates, retain evicted experts.

No quantization or approximate routing. A separate D2H stream backs up GPU-only
records, with explicit readiness for RAM consumers and GPU-slot overwrite.
"""
from collections import OrderedDict
from contextlib import contextmanager
import heapq
from .indexed_store import IndexedExpertStore


class VictimExpertStore(IndexedExpertStore):
    victim_cache = True

    def __init__(self, *a, gpu_only_limit_bytes=None, **kw):
        super().__init__(*a, **kw)
        if not self.pinned:
            raise ValueError('Victim cache requires registered RAM')
        self.gpu_owned = set()
        self.duplicates = OrderedDict()
        self.read_ready = {}
        self.gpu_only_limit = gpu_only_limit_bytes
        self.gpu_only_bytes = 0
        self.largest = max(r['bytes'] for r in self.records.values())
        if self.gpu_only_limit is not None and self.gpu_only_limit < self.largest:
            raise ValueError('GPU-only limit must fit one expert')
        self.frequencies = {}
        self.hot_duplicates = []
        self.stats.update(demotions=0, demote_bytes=0, duplicate_evictions=0)

    def mark_gpu(self, key):
        with self.lock:
            self.gpu_owned.add(key)
            if key in self.cache:
                self.duplicates[key] = None
                self.duplicates.move_to_end(key)
                if self.gpu_only_limit is not None:
                    heapq.heappush(self.hot_duplicates, (-self.frequencies.get(key, 0), key))
            else:
                self.gpu_only_bytes += self.records[key]['bytes']

    def touch_gpu(self, key):
        if self.gpu_only_limit is None:
            return
        with self.lock:
            count = self.frequencies.get(key, 0) + 1
            self.frequencies[key] = count
            if key in self.duplicates:
                heapq.heappush(self.hot_duplicates, (-count, key))
            if len(self.hot_duplicates) > 2 * len(self.gpu_owned) + len(self.records):
                self.hot_duplicates = [(-self.frequencies.get(k, 0), k) for k in self.duplicates]
                heapq.heapify(self.hot_duplicates)

    def _evict(self, recycle_bytes=0):
        key = None
        if self.gpu_only_limit is not None:
            if self.gpu_only_bytes + self.largest <= self.gpu_only_limit:
                while self.hot_duplicates:
                    score, candidate = heapq.heappop(self.hot_duplicates)
                    if candidate in self.duplicates and -score == self.frequencies.get(candidate, 0):
                        key = candidate
                        break
            if key is None:
                key = next((k for k in self.cache if k not in self.gpu_owned), None)
            if key is None:
                raise RuntimeError('RAM and GPU-only budgets cannot retain the GPU cache')
        elif self.duplicates:
            key = next(iter(self.duplicates))
        else:
            key = next(iter(self.cache))
        if key in self.gpu_owned:
            self.gpu_only_bytes += self.records[key]['bytes']
            self.cache.move_to_end(key, last=False)
            self.stats['duplicate_evictions'] += 1
        else:
            self.cache.move_to_end(key, last=False)
        self.duplicates.pop(key, None)
        return super()._evict(recycle_bytes)

    def _wait_dma(self, data):
        super()._wait_dma(data)
        self.read_ready.pop(id(data), None)

    @contextmanager
    def read(self, key, cancel=None):
        with super().read(key, cancel=cancel) as data:
            if key in self.gpu_owned and key not in self.duplicates:
                self.gpu_only_bytes -= self.records[key]['bytes']
                self.duplicates[key] = None
                if self.gpu_only_limit is not None:
                    heapq.heappush(self.hot_duplicates, (-self.frequencies.get(key, 0), key))
            ready = self.read_ready.pop(id(self.cache[key]), None)
            if ready is not None:
                ready.synchronize()
            if key in self.duplicates:
                self.duplicates.move_to_end(key)
            yield data

    def demote(self, key, source, stream):
        """Return the D2H event that must precede overwriting this GPU slot."""
        import torch
        with self.lock:
            if key not in self.gpu_owned:
                raise RuntimeError('GPU ownership lost before demotion')
            ready = None
            if key not in self.cache:
                size = self.records[key]['bytes']
                if source.dtype != torch.uint8 or source.numel() != size or not source.is_cuda:
                    raise ValueError('Invalid victim GPU record')
                self.gpu_only_bytes -= size
                try:
                    data, _ = self._reserve_buffer(size)
                except BaseException:
                    self.gpu_only_bytes += size
                    raise
                target = None
                try:
                    target = torch.frombuffer(data, dtype=torch.uint8)
                    if not target.is_pinned():
                        raise RuntimeError('Victim buffer registration was lost')
                    with torch.cuda.stream(stream):
                        target.copy_(source, non_blocking=True)
                        ready = torch.cuda.Event()
                        ready.record(stream)
                    source.record_stream(stream)
                    self.pending_dma.setdefault(id(data), []).append(ready)
                    self.read_ready[id(data)] = ready
                except BaseException:
                    stream.synchronize()
                    del target
                    target = None
                    self._release(data)
                    self.gpu_only_bytes += size
                    raise
                finally:
                    if target is not None:
                        del target
                self.cache[key] = data
                self.resident += len(data)
                self.stats['peak_resident_bytes'] = max(self.stats['peak_resident_bytes'], self.resident)
                self.stats['demotions'] += 1
                self.stats['demote_bytes'] += size
            self.gpu_owned.remove(key)
            self.duplicates.pop(key, None)
            return ready

    def close(self):
        super().close()
        self.gpu_owned.clear()
        self.duplicates.clear()
        self.read_ready.clear()
        self.gpu_only_bytes = 0
        self.frequencies.clear()
        self.hot_duplicates.clear()
