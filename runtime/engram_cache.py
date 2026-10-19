"""Bounded online cache of original Engram bytes; dequantization is unchanged."""
from collections import OrderedDict
import threading


class RowCache:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError('Row capacity must be positive')
        self.capacity = capacity
        self.rows = OrderedDict()
        self.payload_bytes = 0
        self.lock = threading.RLock()
        self.stats = dict(hits=0, misses=0, evictions=0, peak_rows=0, peak_payload_bytes=0)

    def _touch(self, key):
        self.rows.move_to_end(key)

    def _evict_one(self):
        _, old = self.rows.popitem(last=False)
        self.payload_bytes -= len(old)
        self.stats['evictions'] += 1

    def _insert(self, key, value):
        self.rows[key] = value
        self.payload_bytes += len(value)

    def gather(self, original, handle, ids, nbytes):
        if nbytes <= 0:
            raise ValueError('Row width must be positive')
        with self.lock:
            keys = [(handle, int(row), nbytes) for row in ids]
            # Keep requested bytes locally: a batch can exceed cache capacity.
            found = {}
            missing = []
            for key in keys:
                if key in found:
                    continue
                if key in self.rows:
                    found[key] = self.rows[key]
                    self._touch(key)
                    self.stats['hits'] += 1
                else:
                    missing.append(key)
                    found[key] = None
            if missing:
                blob = original(handle, [key[1] for key in missing], nbytes)
                if len(blob) != len(missing) * nbytes:
                    raise IOError('Incomplete Engram row read')
                self.stats['misses'] += len(missing)
                for i, key in enumerate(missing):
                    value = bytes(blob[i*nbytes:(i+1)*nbytes])
                    found[key] = value
                    while len(self.rows) >= self.capacity:
                        self._evict_one()
                    self._insert(key, value)
                self.stats['peak_rows'] = max(self.stats['peak_rows'], len(self.rows))
                self.stats['peak_payload_bytes'] = max(self.stats['peak_payload_bytes'], self.payload_bytes)
            return b''.join(found[key] for key in keys)

    def clear(self):
        with self.lock:
            self.rows.clear()
            self.payload_bytes = 0


class DecodeRowCache(RowCache):
    """Protect observed generation rows from prefill scans, with fixed capacity.

    Half the entries can be protected. The rest remain available for new rows;
    a decode hit promotes a row, while prefill cannot refresh protected recency.
    Only replacement order changes. Original immutable row bytes are retained.
    """
    def __init__(self, capacity):
        super().__init__(capacity)
        self.protected_capacity = capacity // 2
        self.protected = OrderedDict()
        self.probation = OrderedDict()
        self.phase = 'prefill'

    def set_phase(self, phase):
        if phase not in ('prefill', 'decode'):
            raise ValueError('Unknown Engram cache phase')
        with self.lock:
            self.phase = phase

    def _touch(self, key):
        if self.phase == 'decode' and self.protected_capacity:
            self.probation.pop(key, None)
            self.protected[key] = None
            self.protected.move_to_end(key)
            if len(self.protected) > self.protected_capacity:
                old, _ = self.protected.popitem(last=False)
                self.probation[old] = None
        elif key in self.probation:
            self.probation.move_to_end(key)

    def _insert(self, key, value):
        super()._insert(key, value)
        self.probation[key] = None
        self._touch(key)

    def _evict_one(self):
        # At least one unprotected slot exists, including capacity == 1.
        key, _ = self.probation.popitem(last=False)
        old = self.rows.pop(key)
        self.payload_bytes -= len(old)
        self.stats['evictions'] += 1

    def clear(self):
        with self.lock:
            super().clear()
            self.protected.clear()
            self.probation.clear()


def configure(model, capacity, *, protect_decode=False):
    if protect_decode and capacity == 0:
        raise ValueError('Decode protection requires a positive row capacity')
    if capacity == 0 and not hasattr(model, '_recipe_engram_cache'):
        return None
    from exllamav3.modules.engram import EngramLayer
    cache_type = DecodeRowCache if protect_decode else RowCache
    cache = cache_type(capacity) if capacity else None
    count = 0
    for module in model:
        if not isinstance(module, EngramLayer):
            continue
        count += 1
        table = module.table
        if not hasattr(table, '_recipe_gather_original'):
            table._recipe_gather_original = table._gather
        original = table._recipe_gather_original
        if cache is None:
            table._gather = original
        else:
            # Capture each table's bound reader separately. The cache is shared
            # across both layers, with one hard limit on retained row count.
            table._gather = lambda handle, ids, nbytes, original=original: cache.gather(original, handle, ids, nbytes)
    if count != 2:
        raise ValueError(f'Expected two Engram tables, got {count}')
    model._recipe_engram_cache = cache
    return cache
