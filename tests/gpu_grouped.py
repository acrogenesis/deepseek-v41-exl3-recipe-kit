"""Real K2/K3 expert bytes: grouped projections and MoE versus eager dispatch."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser()
p.add_argument('--allow-gpu', action='store_true')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if not a.allow_gpu:
    raise SystemExit('Explicit GPU authorization required')
a.output.resolve().relative_to((ROOT / 'results').resolve())
if a.output.exists():
    raise FileExistsError(a.output)
import torch
from runtime.cache import ExpertCache
from runtime.indexed_store import IndexedExpertStore, inventory
from runtime.native_dispatch import load_dispatch

torch.set_num_threads(16)
ext = load_dispatch()
records, _ = inventory(ROOT, cached_headers=True)
store = IndexedExpertStore(ROOT / 'work/models/exl3', records, 512*2**20, pinned=True)
capacity = 4*2**30
per = capacity // (40*4096)*4096
quotas = {i: per for i in range(40)}
quotas[39] += capacity-sum(quotas.values())
cache = ExpertCache(store, capacity, layer_quotas=quotas)
result = {'status': 'running', 'cases': [], 'timings': []}

def compare(label, x, y):
    diff = (x-y).abs().max().item()
    row = {'case': label, 'bit_exact': torch.equal(x, y), 'max_error': diff}
    result['cases'].append(row)
    print(json.dumps(row), flush=True)
    assert row['bit_exact'], row

def timed(fn, count=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    t = time.perf_counter()
    for _ in range(count):
        fn()
    end.record()
    end.synchronize()
    return {'wall_us': (time.perf_counter()-t)*1e6/count,
            'device_us': begin.elapsed_time(end)*1000/count}

try:
    with torch.inference_mode():
        for layer in (0, 18):
            ids = [0, 20, 100, 150, 250, 383]
            with ExitStack() as leases:
                projections = [leases.enter_context(cache.lease(f'{layer}:{i}')).projections for i in ids]
                gates = [p['w1'].bc for p in projections]
                ups = [p['w3'].bc for p in projections]
                downs = [p['w2'].bc for p in projections]
                for seed in (17, 1017):
                    torch.manual_seed(seed)
                    for scale in (0.1, 1.0, 10.0):
                        x = torch.randn((1, 5120), device='cuda', dtype=torch.half)*scale
                        for n in (1, 6, 12):
                            ps = (gates+ups)[:n]
                            baseline = torch.cat([p.run_alloc(x, 2304, True) for p in ps])
                            actual = ext.grouped_linear(x, ps)
                            compare(f'{layer}/{seed}/{scale}/gate-up-{n}', baseline, actual)
                        h = torch.randn((6, 2304), device='cuda', dtype=torch.half)*scale
                        expected = torch.cat([p.run_alloc(h[i:i+1], 5120, True) for i,p in enumerate(downs)])
                        compare(f'{layer}/{seed}/{scale}/down', expected, ext.grouped_linear(h, downs))
                        weights = torch.rand((1,6), device='cuda').half()
                        positions = [4,2,5,0,3,1]
                        for limit in (None, 10.0):
                            args = (x, weights, positions, gates, ups, downs, limit)
                            compare(f'{layer}/{seed}/{scale}/moe-{limit}',
                                ext.exact_experts(*args), ext.grouped_experts(*args))
                args = (x, weights, positions, gates, ups, downs, 10.0)
                row = {'layer': layer, 'eager': timed(lambda: ext.exact_experts(*args)),
                       'grouped': timed(lambda: ext.grouped_experts(*args))}
                result['timings'].append(row)
                print(json.dumps(row), flush=True)
            for batch, chosen in enumerate(([1, 21, 101, 151, 251, 382], ids, [2, 22, 102, 152, 252, 381])):
                route = torch.tensor([chosen[::-1]], dtype=torch.long, device='cuda')
                cache.native_dispatch = None
                baseline = cache.apply(layer, x, route, weights)
                cache.native_dispatch = ext
                cache.grouped_dispatch = True
                actual = cache.apply(layer, x, route, weights)
                compare(f'{layer}/slot-reuse-{batch}', baseline, actual)
                cache.native_dispatch = None
            assert cache.stats['evictions'] > 0
    result['gpu_cache'] = dict(cache.stats)
    result['status'] = 'passed'
except BaseException as exc:
    result.update(status='failed', error=repr(exc))
    raise
finally:
    cache.close()
    store.close()
    a.output.write_text(json.dumps(result, indent=2)+'\n')
