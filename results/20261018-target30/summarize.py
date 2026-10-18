"""Summarize measured results; never substitute a warm diagnostic for the suite."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
baseline = json.loads((HERE.parent / '20261018-clean-optimization/optimized-22g.json').read_text())
names = ['larger-ram', 'pinned-84g', 'reused-pinned-84g', 'final-84g', 'selected-84g']
summary = {'target_tok_s': 30, 'target_achieved_on_full_suite': False, 'benchmarks': [], 'diagnostics': {}}
for name in names:
    report = json.loads((HERE / (name + '.json')).read_text())
    assert report['status'] == 'complete', name
    assert len(report['runs']) == len(baseline['runs']) == 14
    identical = all((r['id'], r['phase'], r['token_ids']) == (b['id'], b['phase'], b['token_ids'])
        for r, b in zip(report['runs'], baseline['runs']))
    checks = [r for r in report['runs'] if r['phase'] == 'check']
    assert identical and len(checks) == 8 and all(r['passed'] for r in checks), name
    row = {'name': name, 'fresh_median_tok_s': report['fresh_median_tok_s'],
        'repeat_median_tok_s': report['repeat_median_tok_s'], 'checks_passed': 8,
        'identical_output_tokens': sum(r['output_tokens'] for r in report['runs']),
        'fresh_speedup_vs_previous': report['fresh_median_tok_s'] / baseline['fresh_median_tok_s'],
        'repeat_speedup_vs_previous': report['repeat_median_tok_s'] / baseline['repeat_median_tok_s']}
    if 'cgroup_memory_peak' in report['runs'][0]:
        row.update(cgroup_memory_peak=max(r['cgroup_memory_peak'] for r in report['runs']),
            minimum_sampled_host_available=min(r['host_available_bytes'] for r in report['runs']),
            maximum_cgroup_swap=max(r['cgroup_swap_current'] for r in report['runs']),
            peak_gpu_allocation=max(r['gpu_peak_allocated_bytes'] for r in report['runs']),
            minimum_sampled_gpu_free=min(r['device_free_bytes_after_request'] for r in report['runs']))
    summary['benchmarks'].append(row)
    if name == 'selected-84g':
        for key, source in report['runtime_sources'].items():
            assert hashlib.sha256((ROOT / source['path']).read_bytes()).hexdigest() == source['sha256'], key
        summary['target_achieved_on_full_suite'] = min(report['fresh_median_tok_s'], report['repeat_median_tok_s']) >= 30

for name in ['traffic-v2', 'copy-stream', 'native-decode', 'pinned-trace', 'slots-trace', 'threads']:
    report = json.loads((HERE / name / 'result.json').read_text())
    assert report['status'] == 'complete'
    rows = []
    for i, run in enumerate(report['passes']):
        trace = run['step_trace']
        first, last = trace[0], trace[-1]
        tokens = len(trace) - 1
        rows.append({'pass': i, 'implementation': run['implementation'], 'cpu_threads': run.get('cpu_threads', 16),
            'decode_tok_s': tokens / (last['seconds'] - first['seconds']),
            'expert_ssd_bytes_per_token': (last['ram']['read_bytes'] - first['ram']['read_bytes']) / tokens,
            'expert_h2d_bytes_per_token': (last['h2d_bytes'] - first['h2d_bytes']) / tokens,
            'max_logit_difference': max(run['max_absolute_error_by_position']),
            'argmax_differences': run['same_argmax_as_first'].count(False)})
    summary['diagnostics'][name] = rows

artifacts = []
golden = 'c2ac87f25a1f3c632e76fed7e31586d4b4cccf073f2d58da49b0c91bff28e2e5'
for p in sorted(HERE.glob('*/pass-*-logits.npy')):
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if p.parent.name != 'native-decode' or p.name != 'pass-1-22g-logits.npy':
        assert digest == golden, p
    artifacts.append({'path': str(p.relative_to(HERE)), 'bytes': p.stat().st_size, 'sha256': digest})
summary['logit_artifacts'] = artifacts
for name, count in [('selected-lifetime', 24), ('selected-parity', 8)]:
    test = json.loads((HERE / (name + '.json')).read_text())
    assert test['status'] == 'passed' and len(test['cases']) == count
(HERE / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary['benchmarks'], indent=2))
