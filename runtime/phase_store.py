"""Opt-in victim store that protects experts observed during decode from scans."""
from contextlib import contextmanager
import heapq

from .decode_priority import DecodePriority
from .indexed_store import IndexedExpertStore
from .victim_store import VictimExpertStore


class PhaseExpertStore(VictimExpertStore):
    prioritize_decode = True
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.priority = DecodePriority()
        self.phase = 'prefill'
        self.stats.update(decode_observations=0, priority_evictions=0)

    def set_phase(self, phase):
        if phase not in ('prefill', 'decode', 'transition'):
            raise ValueError('Unknown inference phase')
        with self.lock:
            self.phase = phase

    def touch_gpu(self, key):
        # Both cache frequencies count actual generation accesses. A full expert
        # scan or a resize warm-up cannot masquerade as decode use.
        if self.phase == 'decode':
            super().touch_gpu(key)
            self.priority.observe_decode(key)
            self.stats['decode_observations'] += 1

    @contextmanager
    def read(self, key, cancel=None):
        with super().read(key, cancel=cancel) as data:
            self.priority.admit(key)
            yield data

    def demote(self, key, source, stream):
        event = super().demote(key, source, stream)
        if key in self.cache:
            # It may previously have been popped from the heap while GPU-owned.
            self.priority.admit(key)
        return event

    def _evict(self, recycle_bytes=0):
        key = None
        if self.gpu_only_limit is not None:
            if self.gpu_only_bytes + self.largest <= self.gpu_only_limit:
                while self.hot_duplicates:
                    score, candidate = heapq.heappop(self.hot_duplicates)
                    if candidate in self.duplicates and -score == self.frequencies.get(candidate, 0):
                        key = candidate
                        break
        elif self.duplicates:
            key = next(iter(self.duplicates))
        if key is None:
            key = self.priority.candidate(self.cache, self.gpu_owned)
            self.stats['priority_evictions'] += 1
        if key in self.gpu_owned:
            self.gpu_only_bytes += self.records[key]['bytes']
            self.stats['duplicate_evictions'] += 1
        self.cache.move_to_end(key, last=False)
        self.duplicates.pop(key, None)
        self.priority.forget_ram(key)
        return IndexedExpertStore._evict(self, recycle_bytes)

    def close(self):
        super().close()
        self.priority.clear()
