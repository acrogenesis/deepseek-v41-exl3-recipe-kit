#!/usr/bin/env python3
"""Render the filled-256K benchmark charts from audited receipts (CPU only).

Run with Python and matplotlib 3.10.8: python scripts/plot_context.py
"""
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from audit_context import audit

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/20261019-context256k'
OUT = RESULTS / 'charts'
BG, GRID, INK, MUTED = '#07110d', '#284338', '#e8f3ed', '#a3bcae'
SERIES = (
    ('Initial cache', 'filled-01.json', '#51d6ed'),
    ('Protected caches', 'filled-desktop.json', '#b2ff68'),
)


def measurements():
    report = audit(RESULTS / 'filled-desktop.json', RESULTS / 'filled-fixture.json',
                   RESULTS / 'desktop-snapshot-sources.json', RESULTS / 'filled-01.json')
    if not report['evidence_valid']:
        raise ValueError(report['errors'])
    values = []
    for label, filename, color in SERIES:
        path = RESULTS / filename
        data = json.loads(path.read_text())
        rows = {}
        for row in data['runs']:
            events = row['events']
            seconds = events[-1]['seconds'] - events[0]['seconds']
            rate = (len(row['token_ids']) - len(events[0]['tokens'])) / seconds
            if not math.isclose(rate, row['decode_tok_s'], rel_tol=1e-9):
                raise ValueError(f'Stream timing mismatch: {filename}')
            rows[row['phase']] = {
                'decode_tok_s': rate,
                'ttft_seconds': events[0]['seconds'],
                'input_tokens': row['input_tokens'],
                'output_tokens': row['output_tokens'],
            }
        profile = data['profile']
        values.append(dict(label=label, color=color, rows=rows,
                           source=str(path.relative_to(ROOT)),
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           gpu_gib=profile['gpu_expert_cache_bytes'] / 2**30,
                           ram_gib=profile['ram_expert_cache_bytes'] / 2**30))
    return values


def chart(values, metric, title, unit, filename, divisor=1):
    fig, ax = plt.subplots(figsize=(10, 6.2), facecolor=BG)
    ax.set_facecolor(BG)
    fig.subplots_adjust(left=.22, right=.92, top=.66, bottom=.29)
    maximum = 0
    for i, phase in enumerate(('fresh', 'repeat')):
        ax.axhspan(i - .37, i + .37, color='#0c1b14', zorder=0)
        for j, series in enumerate(values):
            value = series['rows'][phase][metric] / divisor
            y = i + (j - .5) * .26
            color = series['color']
            maximum = max(maximum, value)
            label = f"{series['label']} · {series['gpu_gib']:g}/{series['ram_gib']:g} GiB"
            ax.barh(y, value, height=.18, color=color, alpha=.33,
                    label=label if i == 0 else None, zorder=2)
            ax.plot([value, value], [y - .09, y + .09], color=color, lw=2, zorder=3)
            ax.annotate(f'{value:.2f}', (value, y), xytext=(10, 0),
                        textcoords='offset points', va='center', color=color,
                        fontsize=13, weight='bold')
    ax.set_yticks((0, 1), ('Fresh request', 'Warm repeat'), fontsize=11)
    ax.set_ylim(1.5, -.4)
    ax.set_xlim(0, math.ceil(maximum * 1.18 / 5) * 5)
    ax.xaxis.set_major_locator(MultipleLocator(10 if divisor == 1 else 5))
    ax.xaxis.grid(True, color=GRID, lw=.8, zorder=1)
    ax.tick_params(length=0, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(unit, labelpad=14, fontsize=10)
    legend = ax.legend(loc='lower left', bbox_to_anchor=(-.02, 1.10),
                       ncol=2, frameon=False, fontsize=9, columnspacing=1.6)
    for handle in legend.legend_handles:
        handle.set_alpha(1)
    fig.text(.075, .94, 'FIELD NOTES / RTX 5090 / DEEPSEEK V4.1 FLASH 1017',
             color='#b2ff68', fontsize=10, family='monospace')
    fig.text(.075, .855, title, fontsize=25, weight='bold')
    fig.text(.075, .795, '262,015 input tokens · 128 output tokens · 262,144-token window',
             fontsize=11, color=MUTED)
    fig.text(.075, .14, 'One request per bar. Fresh KV state each time; expert and Engram caches can warm.',
             fontsize=8.7, color=MUTED)
    timing = ('Generation after the first token; prefill excluded.' if divisor == 1 else
              'Includes prefill, cache resize and any enabled verification.')
    fig.text(.075, .105, timing + ' Legend: expert GPU/RAM budgets.', fontsize=8.7, color=MUTED)
    fig.text(.075, .05, 'acrogenesis.com · 19 OCT 2026 · RTX 5090 32 GB / RAM 128 GB',
             fontsize=8.5, color=MUTED, family='monospace')
    for extension in ('png', 'svg'):
        metadata = {'Date': '2026-10-19'} if extension == 'svg' else {}
        path = OUT / f'{filename}.{extension}'
        fig.savefig(path, dpi=180, facecolor=BG, metadata=metadata)
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK,
                         'axes.labelcolor': MUTED, 'xtick.color': MUTED,
                         'ytick.color': INK, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'deepseek-context256k'})
    values = measurements()
    OUT.mkdir(exist_ok=True)
    chart(values, 'decode_tok_s', 'Generation speed',
          'output tokens / second · higher is better', 'generation-speed')
    chart(values, 'ttft_seconds', 'The wait before the answer',
          'minutes to first token · lower is better', 'time-to-first-token', divisor=60)
    (OUT / 'measurements.json').write_text(json.dumps({
        'matplotlib': matplotlib.__version__, 'series': values,
        'context_tokens': 262144, 'requests_per_bar': 1,
        'metric': 'Streamed output tokens after the first token / time after the first token',
        'ttft_display_unit': 'minutes',
    }, indent=2) + '\n')
    print(f'Rendered generation speed and time to first token in {OUT}')


if __name__ == '__main__':
    main()
