"""Teacher-forced cache comparison and CPU profile; excluded from speed results."""
import argparse
import cProfile
import json
import os
from pathlib import Path
import pstats
import time
import hashlib
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-gpu', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--caches', type=int, nargs='+', default=[16, 22, 16])
    parser.add_argument('--tokens', type=int, default=20)
    parser.add_argument('--trace', action='store_true', help='Record expert traffic separately for each token')
    parser.add_argument('--no-profile', action='store_true')
    parser.add_argument('--ram-cache-gib', type=int, choices=range(16, 85))
    parser.add_argument('--pinned-ram', action='store_true')
    parser.add_argument('--threads', type=int, choices=range(1, 33), nargs='+', help='One CPU thread count per pass')
    parser.add_argument('--baseline-cache', type=Path,
        help='Use this older cache.py for the first pass, then the current cache')
    parser.add_argument('--baseline-final', action='store_true', help='Also use the baseline for the last pass')
    parser.add_argument('--reference', type=Path, default=ROOT / 'results/20261017T232708Z-cache-campaign/bench-16g.json')
    args = parser.parse_args()
    if args.threads and len(args.threads) != len(args.caches):
        parser.error('--threads needs one value per cache pass')
    if not args.allow_gpu:
        raise SystemExit('Explicit GPU authorization required')
    args.output.resolve().relative_to((ROOT / 'results').resolve())
    args.output.mkdir(exist_ok=False, parents=True)
    from scripts.preflight import check
    assert check()['ready_for_gpu_validation']
    for name in ('EXL3_ATS_MMAP', 'EXL3_ENGRAM_ATS', 'EXL3_MOE_CPU_SPLIT', 'EXL3_MOE_CPU_OFFLOAD', 'EXL3_INT8_GEMV'):
        os.environ[name] = '0'
    import numpy as np
    import torch
    from exllamav3 import Config, Tokenizer
    from exllamav3.cache.cache import Cache
    from exllamav3.generator import Generator, Job
    from exllamav3.generator.sampler import GreedySampler
    from .indexed_store import inventory, IndexedExpertStore
    from .cache import ExpertCache
    baseline_class = ExpertCache
    if args.baseline_cache:
        spec = importlib.util.spec_from_file_location('runtime.baseline_cache', args.baseline_cache)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        baseline_class = module.ExpertCache
    from .native_adapter import attach_engram, build_model
    profile = json.loads((ROOT / 'profiles/5090.json').read_text())
    if args.ram_cache_gib is not None:
        profile['ram_expert_cache_bytes'] = args.ram_cache_gib * 2**30
    records, _ = inventory(ROOT)
    store = IndexedExpertStore(ROOT / 'work/models/exl3', records, profile['ram_expert_cache_bytes'],
        reserve_bytes=profile['host_reserve_bytes'], pinned=args.pinned_ram)
    fixture = json.loads((ROOT / 'fixtures/paired-gguf.json').read_text())['speed'][0]
    reference = json.loads(args.reference.read_text())
    forced = next(r['token_ids'] for r in reference['runs'] if r['id'] == fixture['id'] and r['phase'] == 'fresh')[:args.tokens]
    assert len(forced) == args.tokens
    result = {'status': 'running', 'diagnostic_only': True, 'case': fixture['id'], 'forced_tokens': forced, 'passes': [],
        'pinned_ram': args.pinned_ram, 'ram_cache_gib': profile['ram_expert_cache_bytes'] / 2**30,
        'runtime_sha256': {name: hashlib.sha256((ROOT / 'runtime' / name).read_bytes()).hexdigest()
            for name in ('cache.py', 'indexed_store.py', 'diagnostic.py')},
        'tune_cache_path': os.environ.get('EXLLAMAV3_TUNE_CACHE')}
    if args.baseline_cache:
        result['baseline_cache_sha256'] = hashlib.sha256(args.baseline_cache.read_bytes()).hexdigest()
    cache = model = None
    first_logits = None

    class CaptureSampler(GreedySampler):
        def __init__(self):
            super().__init__()
            self.rows = []
            self.trace = []

        def forward(self, logits, *a, **kw):
            row = logits.detach().reshape(-1).float().cpu().clone()
            index = len(self.rows)
            self.rows.append(row.numpy())
            if args.trace:
                self.trace.append({'position': index, 'seconds': time.perf_counter(),
                    'gpu': dict(cache.stats), 'ram': dict(store.stats),
                    'h2d_bytes': cache.h2d_bytes, 'requested_experts': cache.requests})
                cache.requests = []
            return torch.tensor([[forced[index]]], device=logits.device, dtype=torch.long)

    def save():
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    try:
        for index, gib in enumerate(args.caches):
            if args.threads:
                torch.set_num_threads(args.threads[index])
            if cache is not None:
                cache.close()
                torch.cuda.empty_cache()
            size = gib * 2**30
            per = size // (40 * 4096) * 4096
            quotas = {i: per for i in range(40)}
            quotas[39] += size - sum(quotas.values())
            is_baseline = bool(args.baseline_cache) and (index == 0 or args.baseline_final and index == len(args.caches) - 1)
            cache_class = baseline_class if is_baseline else ExpertCache
            if args.trace:
                class TracedCache(cache_class):
                    h2d_bytes = 0
                    def __init__(self, *a, **kw):
                        super().__init__(*a, **kw)
                        self.requests = []
                    def _load(self, key, cancel):
                        entry = super()._load(key, cancel)
                        self.h2d_bytes += self.store.records[key]['bytes']
                        return entry
                    def lease(self, key, **kw):
                        self.requests.append(key)
                        return super().lease(key, **kw)
                cache_class = TracedCache
            cache = cache_class(store, size, layer_quotas=quotas, reserve_bytes=profile['host_reserve_bytes'])
            if model is None:
                config = Config.from_directory(str(ROOT / 'work/models/exl3'))
                config.infer_params.no_reconstruct = True
                attach_engram(config, ROOT / 'work/models/engram')
                model = build_model(config, cache)
                kv = Cache(model, max_num_tokens=8192, max_batch_size=1)
                model.load('cuda:0', progressbar=False)
                tokenizer = Tokenizer.from_config(config)
            else:
                for module in model:
                    if hasattr(module, 'expert_cache'):
                        module.expert_cache = cache
            generator = Generator(model=model, cache=kv, tokenizer=tokenizer,
                max_batch_size=1, max_chunk_size=256, recurrent_cache_size=512 * 2**20)
            sampler = CaptureSampler()
            job = Job(input_ids=torch.tensor([fixture['input_ids']], dtype=torch.long),
                max_new_tokens=len(forced), sampler=sampler, stop_conditions=[])
            generator.enqueue(job)
            profiler = cProfile.Profile()
            started = time.perf_counter()
            with torch.inference_mode():
                if not args.no_profile:
                    profiler.enable()
                progress = time.monotonic()
                while generator.num_remaining_jobs():
                    events = generator.iterate()
                    if events:
                        progress = time.monotonic()
                    if time.monotonic() - progress > 120:
                        raise TimeoutError('Generator made no progress for 120 seconds')
                    for event in events:
                        if event.get('stage') == 'error':
                            raise RuntimeError(str(event))
                profiler.disable()
            torch.cuda.synchronize()
            duration = time.perf_counter() - started
            logits = np.stack(sampler.rows)
            assert logits.shape[0] == len(forced)
            assert np.isfinite(logits[:, :config.vocab_size]).all()
            np.save(args.output / f'pass-{index}-{gib}g-logits.npy', logits)
            if not args.no_profile:
                profiler.dump_stats(str(args.output / f'pass-{index}-{gib}g.pstats'))
                with (args.output / f'pass-{index}-{gib}g-profile.txt').open('w') as f:
                    pstats.Stats(profiler, stream=f).sort_stats('cumulative').print_stats(55)
                    pstats.Stats(profiler, stream=f).sort_stats('tottime').print_stats(35)
            row = {'cache_gib': gib, 'implementation': 'baseline' if is_baseline else 'current',
                'cpu_threads': torch.get_num_threads(),
                'seconds': duration, 'ram': dict(store.stats), 'gpu': dict(cache.stats),
                'argmax_tokens': logits.argmax(-1).tolist()}
            if args.trace:
                for step in sampler.trace:
                    step['seconds'] -= started
                row['step_trace'] = sampler.trace
            valid = logits[:, :config.vocab_size]
            if first_logits is None:
                first_logits = valid.copy()
            difference = np.abs(valid - first_logits)
            row['max_absolute_error_by_position'] = difference.max(-1).tolist()
            row['same_argmax_as_first'] = (valid.argmax(-1) == first_logits.argmax(-1)).tolist()
            order = np.argsort(valid, axis=-1)[:, -2:]
            row['top_two_tokens'] = order.tolist()
            row['top_two_margin'] = (np.take_along_axis(valid, order[:, -1:], -1) - np.take_along_axis(valid, order[:, :1], -1)).flatten().tolist()
            result['passes'].append(row)
            save()
            print(json.dumps({'pass': index, 'cache': gib, 'seconds': duration,
                'max_logit_difference': float(difference.max()), 'same_argmax': all(row['same_argmax_as_first'])}), flush=True)
        result['status'] = 'complete'
    except BaseException as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        save()
        if model is not None:
            model.unload()
        if cache is not None:
            cache.close()
        store.close()


if __name__ == '__main__':
    main()
