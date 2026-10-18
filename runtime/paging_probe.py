"""Instrument the full benchmark's paging costs; not a speed benchmark.

Run with the same arguments as runtime.runner. Timing counters are cumulative
and nested, so they must not be added together as independent wall time.
"""
import functools
import os
import sys
import time


def main():
    if '--allow-gpu' not in sys.argv:
        raise SystemExit('Explicit --allow-gpu required')
    from .indexed_store import IndexedExpertStore
    from .cache import ExpertCache
    from . import runner

    active_store = None
    original_read = os.preadv

    def preadv(*a, **kw):
        start = time.perf_counter()
        try:
            return original_read(*a, **kw)
        finally:
            if active_store is not None:
                stats = active_store.stats
                stats['probe_preadv_seconds'] = stats.get('probe_preadv_seconds', 0.0) + time.perf_counter() - start

    def instrument(cls, name):
        original = getattr(cls, name)
        @functools.wraps(original)
        def measured(self, *a, **kw):
            nonlocal active_store
            previous = active_store
            if isinstance(self, IndexedExpertStore):
                active_store = self
            start = time.perf_counter()
            try:
                return original(self, *a, **kw)
            finally:
                elapsed = time.perf_counter() - start
                key = 'probe_' + name.lstrip('_') + '_seconds'
                self.stats[key] = self.stats.get(key, 0.0) + elapsed
                active_store = previous
        setattr(cls, name, measured)

    os.preadv = preadv
    for cls, names in ((IndexedExpertStore, ('_read_extent', '_fd', '_allocate', '_evict', '_wait_dma')),
                       (ExpertCache, ('_guard', '_load', '_evict', 'apply'))):
        for name in names:
            instrument(cls, name)
    print('INSTRUMENTED PAGING DIAGNOSTIC: nested counters; throughput is not a headline benchmark', flush=True)
    runner.main()


if __name__ == '__main__':
    main()
