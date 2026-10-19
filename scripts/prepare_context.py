"""Create an exact-length text fixture with explicit, reproducible retrieval targets.

Uses the already verified official chat wrapper from paired-gguf.json. No GPU.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--context', type=int, default=262144)
    ap.add_argument('--input-tokens', type=int, required=True)
    ap.add_argument('--output-cap', type=int, default=128)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.input_tokens + args.output_cap + 1 > args.context:
        ap.error('Input + output cap + native scheduler position must fit the context')
    if args.output.exists():
        raise FileExistsError(args.output)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ROOT / 'work/models/exl3/tokenizer.json'))
    source = ROOT / 'fixtures/paired-gguf.json'
    base = json.loads(source.read_text())['speed'][0]
    original = base['messages'][0]['content']
    assert base['rendered_prompt'].count(original) == 1
    before, after = base['rendered_prompt'].split(original)
    assert tok.encode(base['rendered_prompt'], add_special_tokens=False).ids == base['input_ids']
    keys = {'cedar': 'R8D2-JUNE', 'maple': 'K3P7-NORTH', 'birch': 'T6W1-SILVER'}
    prefix = 'Review the following deployment records and retain the special registry entries.\n'
    question = ('\nFirst return a JSON object containing the exact registry codes for cedar, maple and birch. '
                'Then explain failure modes and improvements for these systems, with worked examples. '
                'Write at least 1000 words after the JSON object.\n')
    systems = ('job queue', 'database replica', 'object store', 'API gateway', 'backup worker', 'event consumer')
    policies = ('retry with backoff', 'verify the checksum', 'write an audit record', 'check the lease',
                'persist a checkpoint', 'reject duplicate writes', 'refresh the cache')

    def content(n, padding=0):
        inserts = {n // 10: ('cedar', keys['cedar']), n // 2: ('maple', keys['maple']),
                   9 * n // 10: ('birch', keys['birch'])}
        lines = []
        for i in range(n):
            if i in inserts:
                key, code = inserts[i]
                lines.append(f'SPECIAL REGISTRY ENTRY: {key} has code {code}.\n')
            lines.append(f'Record {i}: the {systems[i % len(systems)]} must {policies[i % len(policies)]}. '
                         f'Timeout is {7 + i % 113} seconds; tenant is tenant-{i % 997}; '
                         f'expected revision is {i * 17 + 31}. Record the outcome before acknowledging completion.\n')
        return prefix + ''.join(lines) + ' x' * padding + question

    def encode(text):
        return tok.encode(before + text + after, add_special_tokens=False).ids

    lo, hi = 10, max(11, args.input_tokens // 32 + 32)
    assert len(encode(content(lo))) <= args.input_tokens < len(encode(content(hi)))
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if len(encode(content(mid))) <= args.input_tokens:
            lo = mid
        else:
            hi = mid
    padding = args.input_tokens - len(encode(content(lo)))
    for _ in range(8):
        text = content(lo, padding)
        ids = encode(text)
        difference = args.input_tokens - len(ids)
        if difference == 0:
            break
        padding += difference
        assert padding >= 0
    assert len(ids) == args.input_tokens
    markers = {key: len(tok.encode(before + text[:text.index(f'SPECIAL REGISTRY ENTRY: {key}')],
                                  add_special_tokens=False).ids) for key in keys}
    case = dict(id=f'context-{args.input_tokens}', messages=[dict(role='user', content=text)],
                rendered_prompt=before + text + after, input_ids=ids,
                input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                expected_registry=keys, marker_token_offsets=markers)
    fixture = dict(context=args.context, mode='chat', temperature=0, output_cap=args.output_cap,
                   checks=[], speed=[case], purpose='Filled-context throughput and multi-position retrieval',
                   wrapper_fixture_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   records=lo, padding_tokens=padding)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + '\n')
    print(json.dumps(dict(path=str(args.output), input_tokens=len(ids), markers=markers, records=lo)))


if __name__ == '__main__':
    main()
