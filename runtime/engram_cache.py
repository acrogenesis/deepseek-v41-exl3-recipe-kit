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
                    self.rows.move_to_end(key)
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
                        _, old = self.rows.popitem(last=False)
                        self.payload_bytes -= len(old)
                        self.stats['evictions'] += 1
                    self.rows[key] = value
                    self.payload_bytes += len(value)
                self.stats['peak_rows'] = max(self.stats['peak_rows'], len(self.rows))
                self.stats['peak_payload_bytes'] = max(self.stats['peak_payload_bytes'], self.payload_bytes)
            return b''.join(found[key] for key in keys)

    def clear(self):
        with self.lock:
            self.rows.clear()
            self.payload_bytes = 0


def configure(model, capacity):
    if capacity == 0 and not hasattr(model, '_recipe_engram_cache'):
        return None
    from exllamav3.modules.engram import EngramLayer
    cache = RowCache(capacity) if capacity else None
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
