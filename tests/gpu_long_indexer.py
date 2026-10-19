"""Synthetic kernel gates, not model throughput or model quality measurements."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import MethodType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser()
parser.add_argument('--allow-gpu', action='store_true')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if not args.allow_gpu:
    raise SystemExit('Explicit --allow-gpu required')
if args.output.exists():
    raise FileExistsError(args.output)

import torch
from exllamav3.modules.dsv41 import DSV41Attention
from runtime.long_context import PagedKeys, score_rows, _indexer

torch.manual_seed(1017)
torch.backends.cuda.matmul.allow_tf32 = False
result = dict(status='running', scope='Synthetic indexer correctness and microbenchmark, not model tok/s',
              gpu=torch.cuda.get_device_name(), cases=[])
args.output.parent.mkdir(parents=True, exist_ok=True)


def timed(fn, repeats=10):
    fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repeats


try:
    with torch.inference_mode():
        for ec, seq, rate in [(257, 1, 1), (2048, 17, 2), (20005, 17, 1), (131072, 1, 2), (262144, 1, 1), (262144, 5, 1)]:
            epp = 256 // rate
            pages = (ec + epp - 1) // epp
            pool = torch.randn((pages, epp, 128), dtype=torch.float16, device='cuda')
            bt = torch.randperm(pages, device='cuda').to(torch.int32).unsqueeze(0)
            entries = torch.arange(ec, device='cuda')
            keys = pool[bt[0, entries // epp].long(), entries % epp].contiguous()
            paged = PagedKeys(pool, bt, epp)
            q = torch.randn((seq, 32, 128), dtype=torch.float16, device='cuda')
            w = torch.randn((seq, 32), dtype=torch.float32, device='cuda') / 64
            pos = ec * rate - seq
            visible = entries[None, :] < ((torch.arange(pos, pos + seq, device='cuda') + 1) // rate)[:, None]
            def eager_scores():
                sc = torch.relu(torch.einsum('nhd,ed->nhe', q.float(), keys.float()))
                return torch.einsum('nhe,nh->ne', sc, w).masked_fill(~visible, float('-inf'))
            ref = eager_scores()
            actual = score_rows(q, paged, w, ec, pos, rate)
            error = (actual[visible] - ref[visible]).abs().max().item()
            torch.testing.assert_close(actual, ref, atol=4e-5, rtol=2e-4)
            row = dict(entries=ec, queries=seq, rate=rate, max_score_error=error)
            row['score_eager_ms'] = timed(eager_scores)
            row['score_fused_ms'] = timed(lambda: score_rows(q, paged, w, ec, pos, rate))
            # Exercise projections, RoPE, native candidate selection, causal padding,
            # cross-layer candidate reuse and sorted top-k as the model does.
            identity = SimpleNamespace(forward=lambda value, params: value)
            mod = SimpleNamespace(index_n_heads=32, index_head_dim=128, rope_head_dim=64,
                compress_rate=rate, index_topk=512, ref_quant=False,
                idx_wq_b=identity, idx_weights=identity,
                inv_freq_compress=torch.zeros(32, device='cuda'),
                candidate_topk_blocks=2048, candidate_block_size=8)
            mod._candidate_mask = MethodType(DSV41Attention._candidate_mask, mod)
            qi, xi = q.reshape(1, seq, -1), w.half().unsqueeze(0)
            native_params, fused_params = {}, {}
            row['roles'] = []
            for role in [None, 'source', 'consumer']:
                mod.candidate_role = role
                expected, k = DSV41Attention._v41_indexer(mod, xi, native_params, qi, keys, ec, pos, seq, 0)
                got, got_k = _indexer(mod, xi, fused_params, qi, paged, ec, pos, seq, 0)
                equal = torch.equal(expected, got)
                mismatch = (expected != got).sum().item()
                row['roles'].append(dict(role=role, exact_indices=equal, mismatching_positions=mismatch))
                if role == 'source':
                    row['candidate_mask_equal'] = torch.equal(native_params[('v41_cand', 0)], fused_params[('v41_cand', 0)].dense(ec))
                    assert row['candidate_mask_equal']
                assert k == got_k and equal, row
            result['cases'].append(row)
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(row), flush=True)
        result['status'] = 'passed'
except BaseException as exc:
    result.update(status='failed', error=repr(exc))
    raise
finally:
    args.output.write_text(json.dumps(result, indent=2) + '\n')
