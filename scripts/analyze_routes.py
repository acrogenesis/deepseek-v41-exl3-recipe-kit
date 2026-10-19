"""Summarize observed decode working sets; no model execution or cache preload.

The retrospective top-frequency estimate is diagnostic only. It sees the whole
recorded run and must never be presented as achieved runtime hit rate.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def summarize(run, records, gpu_bytes):
    boundary = run.get('first_token_route_trace_length')
    if boundary is None or 'route_trace' not in run:
        raise ValueError('This run lacks a first-output route boundary')
    trace = run['route_trace'][boundary:]
    counts = Counter()
    layer_frames = Counter()
    for layer, experts in trace:
        layer_frames[layer] += 1
        if len(experts) != 6 or len(set(experts)) != 6:
            raise ValueError('Expected six distinct routed experts per decode layer')
        counts.update(f'{layer}:{expert}' for expert in experts)
    if len(layer_frames) != 40 or len(set(layer_frames.values())) != 1:
        raise ValueError('Incomplete 40-layer decode trace')
    per_layer = gpu_bytes // (40 * 4096) * 4096
    layers = []
    for layer in range(40):
        selected = [key for key in counts if key.startswith(f'{layer}:')]
        quota = per_layer + (gpu_bytes - 40 * per_layer if layer == 39 else 0)
        used = covered = 0
        for key in sorted(selected, key=lambda k: counts[k], reverse=True):
            if used + records[key]['bytes'] <= quota:
                used += records[key]['bytes']
                covered += counts[key]
        layers.append(dict(layer=layer, unique_experts=len(selected),
            unique_bytes=sum(records[key]['bytes'] for key in selected),
            accesses=sum(counts[key] for key in selected),
            retrospective_static_gpu_hits=covered))
    accesses = sum(counts.values())
    delta = run['decode_cache_delta']
    generated = next(iter(layer_frames.values()))
    return dict(id=run['id'], phase=run['phase'], input_tokens=run['input_tokens'],
        measured_decode_steps=generated, decode_tok_s=run['decode_tok_s'],
        unique_experts=len(counts), unique_expert_bytes=sum(records[key]['bytes'] for key in counts),
        expert_accesses=accesses, measured_gpu_hit_fraction=delta['gpu_cache']['hits'] / accesses,
        disk_bytes=delta['ram_cache']['read_bytes'],
        disk_bytes_per_decode_step=delta['ram_cache']['read_bytes'] / generated,
        disk_bytes_per_second_needed_at_30=30 * delta['ram_cache']['read_bytes'] / generated,
        retrospective_static_gpu_hit_fraction=sum(r['retrospective_static_gpu_hits'] for r in layers) / accesses,
        layers=layers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = json.loads(args.result.read_text())
    from runtime.indexed_store import inventory
    records, _ = inventory(ROOT)
    report = dict(source=str(args.result), scope='Observed decode routes; static hit estimate is retrospective, not achieved speed',
        runs=[summarize(run, records, result['profile']['gpu_expert_cache_bytes'])
              for run in result['runs'] if run['phase'] in ('fresh', 'repeat')])
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({**report, 'runs': [{k: v for k, v in run.items() if k != 'layers'} for run in report['runs']]}, indent=2))


if __name__ == '__main__':
    main()
