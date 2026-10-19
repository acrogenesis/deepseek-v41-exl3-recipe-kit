"""Real-weight byte and asynchronous lifetime checks across cache resizing."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument('--allow-gpu', action='store_true')
ap.add_argument('--output', type=Path, required=True)
ap.add_argument('--preserve', action='store_true')
ap.add_argument('--decode-priority', action='store_true')
args = ap.parse_args()
if not args.allow_gpu:
    raise SystemExit('Explicit --allow-gpu required')
if args.output.exists():
    raise FileExistsError(args.output)

import torch
from runtime.cache import ExpertCache, CacheBusy
from runtime.indexed_store import inventory
from runtime.victim_store import VictimExpertStore
if args.decode_priority:
    from runtime.phase_store import PhaseExpertStore as VictimExpertStore
from runtime.elastic_cache import resize

records, _ = inventory(ROOT)
keys = [f'{layer}:{expert}' for layer in (0, 18) for expert in (0, 1)]
records = {k: records[k] for k in keys}
largest = max(r['bytes'] for r in records.values())
store = VictimExpertStore(ROOT / 'work/models/exl3', records, largest * 3,
                         pinned=True, gpu_only_limit_bytes=largest * 4)
cache = ExpertCache(store, largest * 2, layer_quotas={0: largest, 18: largest},
                    policy='lfu' if args.decode_priority else 'lru')
stream = torch.cuda.Stream()
result = dict(status='running', checks=[], resizes=[], preserve=args.preserve, decode_priority=args.decode_priority)
try:
    expected = {}
    for key in keys:
        with store.read(key) as data:
            expected[key] = hashlib.sha256(data).hexdigest()
    for round_, multiplier in enumerate((4, 2, 6, 2)):
        if args.decode_priority:
            # Start with zero-frequency prefill entries: LFU must evict these
            # safely before any generation has trained the frequency tables.
            store.set_phase('prefill' if round_ % 2 == 0 else 'decode')
        snapshots = []
        for key in keys:
            with torch.cuda.stream(stream):
                with cache.lease(key) as entry:
                    if round_ == 0:
                        try:
                            resize(cache, largest * multiplier)
                        except CacheBusy:
                            pass
                        else:
                            raise AssertionError('Resize accepted a live lease')
                    torch.cuda._sleep(1_000_000)
                    snapshot = cache.arena[entry.offset:entry.offset + entry.size].clone()
                    snapshots.append((key, snapshot))
        if args.decode_priority:
            store.set_phase('transition')
        frequency_before = dict(cache.frequencies)
        decode_before = dict(store.priority.counts) if args.decode_priority else None
        result['resizes'].append(resize(cache, largest * multiplier, preserve=args.preserve))
        assert store.gpu_owned == set(cache.entries)
        assert set(store.duplicates) == store.gpu_owned.intersection(store.cache)
        assert store.gpu_only_bytes == sum(records[k]['bytes'] for k in store.gpu_owned if k not in store.cache)
        assert all(cache.frequencies.get(k, 0) == frequency_before.get(k, 0) for k in cache.entries)
        if args.decode_priority:
            assert store.priority.counts == decode_before
        if args.preserve:
            assert cache.entries
            for key, entry in cache.entries.items():
                ready = entry.ready
                ready.synchronize()
                data = cache.arena[entry.offset:entry.offset + entry.size].cpu().numpy().tobytes()
                assert hashlib.sha256(data).hexdigest() == expected[key]
        assert store.resident <= store.capacity
        torch.cuda.synchronize()
        for key, value in snapshots:
            good = hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest() == expected[key]
            result['checks'].append(dict(round=round_, key=key, byte_exact=good))
            assert good
    result['status'] = 'passed'
except BaseException as exc:
    result.update(status='failed', error=repr(exc))
    raise
finally:
    cache.close()
    store.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
