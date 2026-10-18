"""Score the exact held-out tokens and distributions saved by llama-perplexity.

This is a quality measurement, not a speed benchmark. Reference probabilities
use llama.cpp's uint16 encoding and clipped tails; KL uses its > -16 cutoff.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--allow-gpu', action='store_true')
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--corpus', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--gpu-cache-gib', type=int, default=22, choices=range(16, 25))
    ap.add_argument('--chunks', type=int, default=8)
    args = ap.parse_args()
    if not args.allow_gpu:
        ap.error('--allow-gpu is required')
    from scripts.preflight import check
    if not check()['ready_for_gpu_validation']:
        raise RuntimeError('Model staging verification failed')
    args.output.resolve().relative_to((ROOT / 'results').resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('EXL3_ATS_MMAP', 'EXL3_ENGRAM_ATS', 'EXL3_MOE_CPU_SPLIT',
                 'EXL3_MOE_CPU_OFFLOAD', 'EXL3_INT8_GEMV'):
        os.environ[name] = '0'
    import numpy as np
    import torch
    from exllamav3 import Config, Tokenizer
    from exllamav3.cache.cache import Cache
    from exllamav3.generator import Generator, Job
    from exllamav3.generator.sampler import GreedySampler
    from .indexed_store import inventory
    from .victim_store import VictimExpertStore
    from .cache import ExpertCache
    from .native_adapter import attach_engram, build_model
    from .memory_budget import guard_memory
    from .hc_graphs import configure
    from .engram_cache import configure as configure_rows

    with args.reference.open('rb') as f:
        magic, context, vocab, chunks = struct.unpack('<8sIii', f.read(20))
    if magic != b'_logits_' or context != 512 or vocab != 129280 or not 1 <= args.chunks <= chunks:
        raise ValueError((magic, context, vocab, chunks))
    first = context // 2
    count = context - first - 1
    stride = 2 * ((vocab + 1) // 2) + 4
    offset = 20 + chunks * context * 4
    expected_size = offset + chunks * count * stride * 2
    if args.reference.stat().st_size != expected_size:
        raise ValueError('Reference length does not match its header')
    tokens = np.memmap(args.reference, mode='r', dtype='<i4', offset=20, shape=(chunks, context))
    reference = np.memmap(args.reference, mode='r', dtype='<u2', offset=offset,
                          shape=(chunks, count, stride))
    profile = json.loads((ROOT / 'profiles/5090.json').read_text())
    guard_memory(profile['host_reserve_bytes'])
    config = Config.from_directory(str(ROOT / 'work/models/exl3'))
    config.infer_params.no_reconstruct = True
    attach_engram(config, ROOT / 'work/models/engram')
    tokenizer = Tokenizer.from_config(config)
    encoded = tokenizer.encode(args.corpus.read_text(), add_bos=False,
                               encode_special_tokens=True).flatten().tolist()
    if encoded[:chunks * context] != tokens.reshape(-1).tolist():
        raise RuntimeError('Corpus token IDs differ from the GGUF reference')
    # The saved reference has no automatic BOS insertion. Check that explicitly.
    tc = json.loads((ROOT / 'work/models/exl3/tokenizer_config.json').read_text())
    if tc.get('add_bos_token'):
        raise RuntimeError('This fixture requires add_bos_token=false')
    result = {'status': 'running', 'reference_sha256': sha256(args.reference),
              'corpus_sha256': sha256(args.corpus), 'context': context,
              'scored_positions': count * args.chunks, 'chunks': [],
              'gpu_cache_gib': args.gpu_cache_gib, 'ram_cache_gib': 84,
              'mode': 'teacher-forced decode after 257 input tokens; fresh state per chunk',
              'reference_format': 'llama.cpp uint16 log probabilities, KL tails <= -16 omitted',
              'runtime_sha256': {p.name: sha256(p) for p in (ROOT / 'runtime').iterdir()
                                 if p.suffix in ('.py', '.cu', '.cpp', '.h')}}

    def save():
        path = args.output / 'quality.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(result, indent=2) + '\n')
        temp.replace(path)

    save()
    records, _ = inventory(ROOT)
    store = VictimExpertStore(ROOT / 'work/models/exl3', records, 84 * 2**30,
                             reserve_bytes=profile['host_reserve_bytes'], pinned=True,
                             gpu_only_limit_bytes=min(20, args.gpu_cache_gib - 4) * 2**30)
    size = args.gpu_cache_gib * 2**30
    per = size // (40 * 4096) * 4096
    quotas = {i: per for i in range(40)}
    quotas[39] += size - sum(quotas.values())
    experts = ExpertCache(store, size, layer_quotas=quotas,
                         reserve_bytes=profile['host_reserve_bytes'],
                         grouped_dispatch=True, policy='lfu', fast_guard=True,
                         fused_reduce=True, overlap_loads=True, overlap_shared=True)
    model = None
    all_rows = []
    try:
        model = build_model(config, experts)
        kv = Cache(model, max_num_tokens=8192, max_batch_size=1)
        model.load('cuda:0', progressbar=False)
        configure(model, True)
        row_cache = configure_rows(model, 65536)
        for chunk in range(args.chunks):
            forced = tokens[chunk, first + 1:].tolist()
            rows = []
            raw = np.lib.format.open_memmap(args.output / f'chunk-{chunk}-logits.npy',
                                            mode='w+', dtype=np.float32, shape=(count, vocab))

            class Capture(GreedySampler):
                def forward(self, logits, *a, **kw):
                    i = len(rows)
                    x = logits.detach().reshape(-1)[:vocab].float().cpu().numpy()
                    if not np.isfinite(x).all():
                        raise RuntimeError('Non-finite model scores')
                    raw[i] = x
                    x = x.astype(np.float64)
                    logp = x - (x.max() + np.log(np.exp(x - x.max()).sum()))
                    packed = reference[chunk, i]
                    scale, minimum = np.frombuffer(packed[:4].tobytes(), dtype='<f4')
                    base = float(scale) * packed[4:4 + vocab].astype(np.float64) + float(minimum)
                    valid = base > -16
                    target = forced[i]
                    rows.append({'chunk': chunk, 'position': first + i,
                                 'target': target, 'nll': float(-logp[target]),
                                 'reference_nll': float(-base[target]),
                                 'kl': float((np.exp(base[valid]) * (base[valid] - logp[valid])).sum()),
                                 'top1_equal': bool(x.argmax() == base.argmax())})
                    return torch.tensor([[target]], device=logits.device, dtype=torch.long)

            generator = Generator(model=model, cache=kv, tokenizer=tokenizer,
                                  max_batch_size=1, max_chunk_size=256,
                                  recurrent_cache_size=512 * 2**20)
            job = Job(input_ids=torch.tensor([tokens[chunk, :first + 1].tolist()]),
                      max_new_tokens=count, sampler=Capture(), stop_conditions=[])
            generator.enqueue(job)
            start = time.monotonic()
            progress = start
            previous = -1
            with torch.inference_mode():
                while generator.num_remaining_jobs():
                    for event in generator.iterate():
                        if event.get('stage') == 'error':
                            raise RuntimeError(str(event))
                    if len(rows) != previous:
                        progress, previous = time.monotonic(), len(rows)
                    if time.monotonic() - progress > 300:
                        raise TimeoutError('No scored-token progress for five minutes')
            if len(rows) != count:
                raise RuntimeError(f'Expected {count} scores, got {len(rows)}')
            raw.flush()
            del raw
            all_rows.extend(rows)
            summary = {'chunk': chunk, 'seconds': time.monotonic() - start,
                       'positions': len(rows), 'ppl': math.exp(np.mean([r['nll'] for r in rows])),
                       'mean_kl': float(np.mean([r['kl'] for r in rows])),
                       'gpu_free_bytes': torch.cuda.mem_get_info()[0]}
            result['chunks'].append(summary)
            save()
            print(json.dumps(summary), flush=True)
        result.update(status='complete',
                      ppl=math.exp(np.mean([r['nll'] for r in all_rows])),
                      reference_ppl=math.exp(np.mean([r['reference_nll'] for r in all_rows])),
                      mean_kl=float(np.mean([r['kl'] for r in all_rows])),
                      top1_agreement_percent=100 * np.mean([r['top1_equal'] for r in all_rows]),
                      rows=all_rows)
        result['ppl_ratio'] = result['ppl'] / result['reference_ppl']
        result['peak_torch_bytes'] = torch.cuda.max_memory_allocated()
        result['engram_cache'] = dict(row_cache.stats)
    except BaseException as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        save()
        if model is not None:
            model.unload()
        experts.close()
        store.close()


if __name__ == '__main__':
    main()
