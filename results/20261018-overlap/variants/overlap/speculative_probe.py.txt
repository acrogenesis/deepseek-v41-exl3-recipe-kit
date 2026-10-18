"""Bounded DSpark experiment; never changes the default generation path."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-gpu', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.allow_gpu:
        raise SystemExit('Explicit GPU authorization required')
    args.output.resolve().relative_to((ROOT / 'results').resolve())
    if args.output.exists():
        raise FileExistsError(args.output)
    from scripts.preflight import check
    assert check()['ready_for_gpu_validation']
    for name in ('EXL3_ATS_MMAP', 'EXL3_ENGRAM_ATS', 'EXL3_MOE_CPU_SPLIT', 'EXL3_MOE_CPU_OFFLOAD', 'EXL3_INT8_GEMV'):
        os.environ[name] = '0'
    import torch
    from exllamav3 import Config, Model, Tokenizer
    from exllamav3.cache.cache import Cache
    from exllamav3.generator import Generator, Job
    from exllamav3.generator.sampler import GreedySampler
    from .indexed_store import inventory, IndexedExpertStore
    from .cache import ExpertCache
    from .native_adapter import attach_engram, build_model

    records, _ = inventory(ROOT)
    store = IndexedExpertStore(ROOT / 'work/models/exl3', records, 84 * 2**30)
    capacity = 14 * 2**30
    per = capacity // (40 * 4096) * 4096
    quotas = {i: per for i in range(40)}
    quotas[39] += capacity - sum(quotas.values())
    experts = ExpertCache(store, capacity, layer_quotas=quotas, reserve_bytes=12 * 2**30)
    result = {'status': 'loading', 'experimental': True, 'gpu_cache_gib': 14, 'ram_cache_gib': 84,
        'context_tokens': 8192, 'output_tokens': 64, 'runs': [],
        'runtime_sha256': {name: hashlib.sha256((ROOT / 'runtime' / name).read_bytes()).hexdigest()
            for name in ('cache.py', 'indexed_store.py', 'speculative_probe.py')}}
    model = draft = None

    def save():
        args.output.write_text(json.dumps(result, indent=2) + '\n')

    try:
        config = Config.from_directory(str(ROOT / 'work/models/exl3'))
        config.infer_params.no_reconstruct = True
        attach_engram(config, ROOT / 'work/models/engram')
        model = build_model(config, experts)
        draft_config = Config.from_directory(str(ROOT / 'work/models/exl3'))
        draft = Model.from_config(draft_config, component='mtp')
        window = draft.caps['default_draft_size']
        kv = Cache(model, max_num_tokens=8192, max_batch_size=1, max_history=window)
        draft_kv = Cache(draft, max_num_tokens=8192, max_batch_size=1)
        model.load('cuda:0', progressbar=False)
        draft.load('cuda:0', progressbar=False)
        tokenizer = Tokenizer.from_config(config)
        result['draft_window'] = window
        result['allocated_after_load'] = torch.cuda.memory_allocated()
        result['status'] = 'running'
        save()
        print('Target and DSpark loaded', flush=True)
        case = json.loads((ROOT / 'fixtures/paired-gguf.json').read_text())['speed'][0]
        for phase, use_draft in [('cold-baseline', False), ('warm-baseline', False), ('warm-draft', True), ('warm-baseline-again', False)]:
            options = dict(draft_model=draft, draft_cache=draft_kv, record_draft_stats=True) if use_draft else {}
            generator = Generator(model=model, cache=kv, tokenizer=tokenizer, max_batch_size=1,
                max_chunk_size=256, recurrent_cache_size=512 * 2**20, **options)
            job = Job(input_ids=torch.tensor([case['input_ids']], dtype=torch.long), max_new_tokens=64,
                sampler=GreedySampler(), stop_conditions=[])
            generator.enqueue(job)
            tokens, events = [], []
            first = last = None
            first_count = 0
            accepted = rejected = 0
            started = progress = time.perf_counter()
            with torch.inference_mode():
                while generator.num_remaining_jobs():
                    for event in generator.iterate():
                        if event.get('stage') == 'error':
                            raise RuntimeError(str(event))
                        if event.get('stage') != 'streaming':
                            continue
                        accepted = event.get('accepted_draft_tokens', accepted)
                        rejected = event.get('rejected_draft_tokens', rejected)
                        ids = event.get('token_ids')
                        if ids is None or not ids.numel():
                            continue
                        torch.cuda.synchronize()
                        now = time.perf_counter()
                        progress = now
                        if first is None:
                            first, first_count = now, ids.numel()
                        last = now
                        ids = ids.flatten().tolist()
                        tokens.extend(ids)
                        events.append({'seconds': now - started, 'token_ids': ids})
                    if time.perf_counter() - progress > 120:
                        raise TimeoutError('No streamed progress for 120 seconds')
            baseline = result['runs'][0]['token_ids'] if result['runs'] else tokens
            row = {'phase': phase, 'token_ids': tokens, 'events': events,
                'decode_tok_s': (len(tokens) - first_count) / (last - first),
                'ttft_seconds': first - started, 'identical_to_baseline': tokens == baseline,
                'first_difference': next((i for i, (a, b) in enumerate(zip(tokens, baseline)) if a != b), None),
                'accepted_draft_tokens': accepted, 'rejected_draft_tokens': rejected,
                'draft_stats': getattr(job, 'draft_stats', None),
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'free_device_bytes': torch.cuda.mem_get_info()[0],
                'gpu_cache': dict(experts.stats), 'ram_cache': dict(store.stats)}
            result['runs'].append(row)
            save()
            print(json.dumps({k: row[k] for k in ('phase', 'decode_tok_s', 'identical_to_baseline', 'first_difference', 'accepted_draft_tokens', 'rejected_draft_tokens')}), flush=True)
        result['status'] = 'complete'
    except BaseException as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        save()
        if draft is not None:
            draft.unload()
        if model is not None:
            model.unload()
        experts.close()
        store.close()


if __name__ == '__main__':
    main()
