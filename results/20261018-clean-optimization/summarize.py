"""Recompute the summary from the three completed paired benchmark reports."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
NAMES = ['baseline-fixed-16g', 'optimized-16g', 'optimized-22g']
reports = {name: json.loads((ROOT / (name + '.json')).read_text()) for name in NAMES}
baseline = reports[NAMES[0]]
result = {'hardware': baseline['profile']['hardware'], 'metric': baseline['metric'],
    'context_tokens': baseline['profile']['context_tokens'], 'runs_per_phase': 3,
    'concurrent_upload': False, 'implementations': {}}
for name, report in reports.items():
    assert report['status'] == 'complete', (name, report['status'])
    assert report['fixture_sha256'] == baseline['fixture_sha256']
    assert report['inventory']['revision'] == baseline['inventory']['revision']
    assert report['profile']['context_tokens'] == baseline['profile']['context_tokens']
    assert len(report['runs']) == len(baseline['runs']) == 14
    checks = [r for r in report['runs'] if r['phase'] == 'check']
    differences = []
    for expected, actual in zip(baseline['runs'], report['runs']):
        assert (expected['id'], expected['phase']) == (actual['id'], actual['phase'])
        if expected['token_ids'] != actual['token_ids']:
            first = next((i for i, (a, b) in enumerate(zip(expected['token_ids'], actual['token_ids'])) if a != b),
                min(len(expected['token_ids']), len(actual['token_ids'])))
            differences.append({'id': actual['id'], 'phase': actual['phase'], 'first_difference': first})
    row = {'cache_gib': report['profile']['gpu_expert_cache_bytes'] / 2**30,
        'checks_passed': sum(r['passed'] for r in checks), 'checks_total': len(checks),
        'identical_to_baseline': not differences, 'token_differences': differences,
        'generated_tokens_compared': sum(len(r['token_ids']) for r in report['runs']),
        'peak_torch_allocated_gib': max(r['gpu_peak_allocated_bytes'] for r in report['runs']) / 2**30,
        'minimum_device_free_gib': min(r['device_free_bytes_after_request'] for r in report['runs']) / 2**30,
        'tune_cache_sha256_after': report['tune_cache_sha256_after'],
        'report_sha256': hashlib.sha256((ROOT / (name + '.json')).read_bytes()).hexdigest()}
    for phase in ('fresh', 'repeat'):
        runs = [r for r in report['runs'] if r['phase'] == phase]
        assert len(runs) == 3 and all(r['output_tokens'] == 128 for r in runs)
        rates = [r['decode_tok_s'] for r in runs]
        median = statistics.median(rates)
        assert median == report[phase + '_median_tok_s']
        row[phase] = {'tok_s': rates, 'median_tok_s': median,
            'speed_ratio': median / baseline[phase + '_median_tok_s'],
            'median_ttft_seconds': statistics.median(r['ttft_seconds'] for r in runs)}
    result['implementations'][name] = row
result['same_tuning_after_all_runs'] = len({r['tune_cache_sha256_after'] for r in reports.values()}) == 1
(ROOT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
