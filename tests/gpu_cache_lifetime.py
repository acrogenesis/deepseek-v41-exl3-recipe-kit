"""Exercise pinned staging reuse, cross-stream reads, and eviction using real bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser()
parser.add_argument('--allow-gpu', action='store_true')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--pinned-ram', action='store_true')
parser.add_argument('--gpu-policy', choices=('lru', 'lfu'), default='lru')
args = parser.parse_args()
if not args.allow_gpu:
    raise SystemExit('Explicit GPU authorization required')
args.output.resolve().relative_to((ROOT / 'results').resolve())
if args.output.exists():
    raise FileExistsError(args.output)

import torch
from runtime.cache import ExpertCache
from runtime.indexed_store import inventory, IndexedExpertStore

records, _ = inventory(ROOT)
keys = [f'{layer}:{expert}' for layer in (0, 18) for expert in (0, 1)]
records = {key: records[key] for key in keys}
store = IndexedExpertStore(ROOT / 'work/models/exl3', records,
    2 * max(r['bytes'] for r in records.values()), pinned=args.pinned_ram)
quotas = {layer: max(r['bytes'] for key, r in records.items() if key.startswith(f'{layer}:')) for layer in (0, 18)}
cache = ExpertCache(store, sum(quotas.values()), layer_quotas=quotas, policy=args.gpu_policy)
streams = [torch.cuda.Stream() for _ in range(2)]
copies, expected = [], {}
result = {'status': 'running', 'cases': [], 'gpu': torch.cuda.get_device_name(), 'pinned_ram': args.pinned_ram}
result['gpu_policy'] = args.gpu_policy
try:
    for key in keys:
        with store.read(key) as data:
            expected[key] = hashlib.sha256(data).hexdigest()
    for i in range(16):
        # Repeat each expert on the other stream, then evict it. Leaving the
        # stream context before lease release also checks the owning-stream rule.
        key = keys[(i // 2) % len(keys)]
        with torch.cuda.stream(streams[i % 2]):
            lease = cache.lease(key)
            entry = lease.__enter__()
            torch.cuda._sleep(1_000_000)
            snapshot = cache.arena[entry.offset:entry.offset + entry.size].clone()
        lease.__exit__(None, None, None)
        copies.append((key, snapshot))
    cache.clear()
    torch.cuda.synchronize()
    for key, snapshot in copies:
        digest = hashlib.sha256(snapshot.cpu().numpy().tobytes()).hexdigest()
        result['cases'].append({'expert': key, 'byte_exact': digest == expected[key]})
    if args.pinned_ram:
        # Force RAM eviction while a deliberately delayed DMA still owns its
        # source, independently of the GPU expert cache's eviction barriers.
        dma_store = IndexedExpertStore(ROOT / 'work/models/exl3', records,
            max(r['bytes'] for r in records.values()), pinned=True)
        try:
            snapshots = []
            for i in range(8):
                key = keys[i % len(keys)]
                with dma_store.read(key) as data:
                    source = torch.frombuffer(data, dtype=torch.uint8)
                    with torch.cuda.stream(streams[i % 2]):
                        torch.cuda._sleep(20_000_000)
                        snapshot = torch.empty_like(source, device='cuda:0')
                        snapshot.copy_(source, non_blocking=True)
                        ready = torch.cuda.Event()
                        ready.record()
                    dma_store.protect_dma(key, ready)
                    del source
                snapshots.append((key, snapshot))
            dma_store.close()
            for key, snapshot in snapshots:
                digest = hashlib.sha256(snapshot.cpu().numpy().tobytes()).hexdigest()
                result['cases'].append({'expert': key, 'ram_eviction_during_dma': True,
                    'byte_exact': digest == expected[key]})
            assert not dma_store.registered and not dma_store.pending_dma
            from unittest.mock import patch
            with patch.object(dma_store, '_read_extent', side_effect=InterruptedError('test cancellation')):
                try:
                    with dma_store.read(keys[0]):
                        raise AssertionError('Expected interrupted read')
                except InterruptedError:
                    pass
            assert not dma_store.registered and not dma_store.cache
            result['registered_read_failure_cleanup'] = True
        finally:
            dma_store.close()
    assert all(case['byte_exact'] for case in result['cases'])
    result.update(status='passed', cache=dict(cache.stats), ram_cache=dict(store.stats))
finally:
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    cache.close()
    store.close()
