"""Audit a completed filled-context receipt without running inference.

Passing means the recorded evidence is internally consistent and covers the
full window. The measured rate is reported separately; this does not decide
whether a performance goal is achieved.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(result, fixture, snapshot, baseline=None):
    data = json.loads(result.read_text())
    prompt = json.loads(fixture.read_text())
    manifest = json.loads(snapshot.read_text())
    reference = json.loads(baseline.read_text()) if baseline is not None else None
    errors = []

    def check(condition, label):
        if not condition:
            errors.append(label)

    check(data.get('status') == 'complete', 'Benchmark status must be complete')
    check(data['profile']['context_tokens'] == 262144, 'KV capacity must be 262144')
    check(data.get('fixture_sha256') == digest(fixture), 'Fixture hash must match')
    check(reference is not None, 'Paired full-context baseline required')
    if reference is not None:
        check(reference.get('status') == 'complete' and reference.get('fixture_sha256') == digest(fixture),
              'Baseline must be complete and use the same full-context fixture')
    previous = {(row['id'], row['phase']): row for row in reference['runs']} if reference else {}
    check(prompt['context'] == 262144 and prompt['output_cap'] == 128,
          'Fixture must reserve 128 outputs within 256K')
    check(not prompt['checks'] and len(prompt['speed']) == 1, 'Expected one full-window case')
    case = prompt['speed'][0]
    check(len(case['input_ids']) == 262015, 'Fixture must contain 262015 input tokens')
    check(data.get('verify_indexer_first_token') is True, 'Actual native-indexer verification required')
    for name, value in data.get('runtime_source_sha256', {}).items():
        check(manifest.get('runtime/' + name) == value, 'Runtime snapshot mismatch: ' + name)
    expected_runtime = {name.removeprefix('runtime/') for name in manifest
                        if name.startswith('runtime/') and '/' not in name.removeprefix('runtime/')
                        and Path(name).suffix in ('.py', '.cpp', '.cu', '.h')}
    check(set(data.get('runtime_source_sha256', {})) == expected_runtime, 'Runtime source list must be complete')
    model_lock = json.loads((ROOT / 'models.lock.json').read_text())
    check(manifest.get('models.lock.json') == digest(ROOT / 'models.lock.json'),
          'Model lock must match the frozen snapshot')
    model = next(item for item in model_lock['models'] if item['id'] == 'exl3')
    check(data['inventory']['model'] == model['repo'] and data['inventory']['revision'] == model['revision'],
          'EXL3 model identity must match the pinned input')
    check(data['runtime_lock'] == json.loads((ROOT / 'runtime.lock.json').read_text()),
          'Native runtime lock must match')

    native = data.get('indexer_verification', [])
    check(len(native) == 8 and {item['layer'] for item in native} == {2, 8, 14, 20, 24, 28, 32, 36},
          'All eight native indexers must be checked')
    for item in native:
        check(item['position'] == 262014 and item['entries'] == (131007 if item['layer'] < 20 else 262015),
              'Native indexer must use the filled prompt: layer ' + str(item['layer']))
        check(item['indices_equal'] and item.get('candidate_mask_equal', True) and item['differing_positions'] == 0,
              'Native index selections differ: layer ' + str(item['layer']))
        check(item['k'] == 512, 'Native top-k size must remain 512')
        if item['layer'] == 20:
            check(item.get('candidate_mask_equal') is True, 'Source candidate mask comparison is required')

    check(len(data['runs']) == 2 and {row['phase'] for row in data['runs']} == {'fresh', 'repeat'},
          'Both fresh and repeated requests must complete')
    rows = []
    for row in data['runs']:
        phase = row['phase']
        prior = previous.get((row['id'], phase))
        token_parity = prior is not None and row['token_ids'] == prior['token_ids']
        check(token_parity, phase + ': output tokens must match the full-context baseline')
        check(row['id'] == case['id'] and row['input_tokens'] == 262015 and row['output_tokens'] == 128,
              phase + ': full input and output sizes required')
        check(row['final_kv_positions'] == [262142], phase + ': final KV position must cover the full prompt')
        events = row['events']
        check([token for event in events for token in event['tokens']] == row['token_ids'],
              phase + ': streamed tokens must match recorded output')
        check(len(row['token_ids']) == 128, phase + ': require 128 actual output tokens')
        times = [event['seconds'] for event in events]
        check(all(math.isfinite(t) for t in times) and all(a <= b for a, b in zip(times, times[1:])),
              phase + ': stream timings must be finite and ordered')
        elapsed = times[-1] - times[0]
        rate = (len(row['token_ids']) - len(events[0]['tokens'])) / elapsed if elapsed > 0 else 0
        check(math.isclose(rate, row['decode_tok_s'], rel_tol=1e-9), phase + ': rate must match streamed timings')
        check(math.isclose(times[0], row['ttft_seconds'], rel_tol=1e-9), phase + ': TTFT must match first event')
        check(row['retrieval']['passed'] and row['retrieval']['actual'] == case['expected_registry'],
              phase + ': all three retrieval codes must be correct')
        check(row['cgroup_swap_current'] == 0, phase + ': model must not swap')
        check(math.isclose(data[phase + '_median_tok_s'], rate, rel_tol=1e-9), phase + ': summary rate must match')
        rows.append(dict(phase=phase, input_tokens=row['input_tokens'], output_tokens=row['output_tokens'],
            decode_tok_s=rate, difference_from_30_percent=(rate / 30 - 1) * 100, baseline_tokens_equal=token_parity,
            ttft_seconds=row['ttft_seconds'], expert_disk_bytes=row['decode_cache_delta']['ram_cache']['read_bytes'],
            engram=row['decode_cache_delta'].get('engram_cache'), host_available_bytes=row['host_available_bytes']))
    return dict(evidence_valid=not errors, scope='Receipt audit; throughput goal must be assessed separately',
        source=str(result), source_sha256=digest(result), fixture_sha256=digest(fixture),
        baseline_sha256=digest(baseline) if baseline is not None else None,
        snapshot_manifest_sha256=digest(snapshot), errors=errors, measurements=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--snapshot-manifest', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(args.result, args.fixture, args.snapshot_manifest, args.baseline)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['evidence_valid']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
