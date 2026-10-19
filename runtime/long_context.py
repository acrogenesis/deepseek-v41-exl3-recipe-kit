"""Opt-in DSV4.1 index scoring without the query x head x history temporary.

The checkpoint, KV precision, causal mask, candidate blocks and top-k stay fixed.
Queries and keys are native FP16 values with FP32 dot-product accumulation.
The reduction order can differ from eager torch; GPU parity gates quantify it.
"""
from dataclasses import dataclass
from types import MethodType

import torch
import triton
import triton.language as tl


@dataclass
class PagedKeys:
    pool: torch.Tensor
    table: torch.Tensor
    entries_per_page: int


@dataclass
class CandidateBlocks:
    blocks: torch.Tensor
    block_size: int

    def dense(self, ec):
        """Reference view for validation only; the runtime keeps block IDs."""
        entries = torch.arange(ec, device=self.blocks.device)
        keep = torch.zeros((self.blocks.shape[0], triton.cdiv(ec, self.block_size) + 1),
                           dtype=torch.bool, device=self.blocks.device)
        keep.scatter_(1, torch.where(self.blocks >= 0, self.blocks, keep.shape[1] - 1).long(), True)
        return keep[:, entries // self.block_size]


def candidate_blocks(scores, comp_lens, ec, block_size, topk):
    n = triton.cdiv(ec, block_size)
    padded = torch.full((scores.shape[0], n * block_size), float('-inf'), device=scores.device)
    padded[:, :ec] = scores
    blocks = padded.view(scores.shape[0], n, block_size).amax(-1)
    newest = (comp_lens - 1).clamp_min(0) // block_size
    blocks[torch.arange(scores.shape[0], device=scores.device), newest] = float('inf')
    top = blocks.topk(min(topk, n), dim=-1)
    return torch.where(top.values > float('-inf'), top.indices, -1).to(torch.int32)


@triton.jit(do_not_specialize=['EC', 'POS', 'START'])
def _score_kernel(Q, K, W, BT, CAND, OUT, EC, POS, START,
                  RATE: tl.constexpr, EPP: tl.constexpr, PAGED: tl.constexpr,
                  CAND_COUNT: tl.constexpr, CAND_SIZE: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(1)
    entries = tl.program_id(0) * BN + tl.arange(0, BN)
    if CAND_COUNT:
        block = tl.load(CAND + (row + START) * CAND_COUNT + entries // CAND_SIZE,
                        entries < CAND_COUNT * CAND_SIZE, other=-1)
        entries = block * CAND_SIZE + entries % CAND_SIZE
    valid = (entries >= 0) & (entries < EC)
    heads = tl.arange(0, 32)
    dims = tl.arange(0, 128)
    q = tl.load(Q + (row + START) * 4096 + heads[:, None] * 128 + dims[None, :])
    if PAGED:
        page = tl.load(BT + entries // EPP, valid, other=0)
        physical = page * EPP + entries % EPP
    else:
        physical = entries
    k = tl.load(K + dims[:, None] + physical[None, :] * 128,
                valid[None, :], other=0)
    dot = tl.dot(q, k, out_dtype=tl.float32)
    weights = tl.load(W + (row + START) * 32 + heads)
    score = tl.sum(tl.maximum(dot, 0.0) * weights[:, None], axis=0)
    visible = entries < (POS + START + row + 1) // RATE
    score = tl.where(visible, score, float('-inf'))
    tl.store(OUT + row * EC + entries, score, valid)


def score_rows(q, keys, weights, ec, pos0, rate, start=0, count=None, candidate=None):
    """Bounded caller-selected query tile; accepts real paged or contiguous keys."""
    count = q.shape[0] - start if count is None else count
    if q.dtype != torch.float16 or q.shape[1:] != (32, 128) or not q.is_contiguous():
        raise ValueError('Indexer requires contiguous FP16 queries with 32 heads x 128 dimensions')
    paged = isinstance(keys, PagedKeys)
    pool = keys.pool if paged else keys
    if pool.dtype != torch.float16 or not pool.is_contiguous():
        raise ValueError('Indexer keys must be contiguous FP16')
    if not weights.is_contiguous() or (candidate is not None and not candidate.blocks.is_contiguous()):
        raise ValueError('Weights and candidate mask must be contiguous')
    # Scatter the compact computation back into the original score layout. This
    # preserves torch.topk's native tie behavior and caps score storage at 32 MiB.
    scores = (torch.full((count, ec), float('-inf'), device=q.device, dtype=torch.float32)
              if candidate is not None else torch.empty((count, ec), device=q.device, dtype=torch.float32))
    nblocks = candidate.blocks.shape[1] if candidate is not None else 0
    bsize = candidate.block_size if candidate is not None else 0
    columns = nblocks * bsize if candidate is not None else ec
    _score_kernel[(triton.cdiv(columns, 128), count)](
        q, pool, weights, keys.table if paged else pool,
        candidate.blocks if candidate is not None else pool, scores,
        ec, pos0, start, rate, keys.entries_per_page if paged else 256, paged,
        nblocks, bsize, 128, num_warps=4, enable_fp_fusion=False)
    return scores


def _gather(self, kl, bt_row, ec):
    return PagedKeys(kl.pool_idx, bt_row, kl.epp)


def _indexer(self, x, params, q_res, keys, ec, pos0, seq, inst):
    from exllamav3.modules.dsv41 import fp4_qdq
    from exllamav3.modules.dsv4 import _ext_rope
    hi, di, rd = self.index_n_heads, self.index_head_dim, self.rope_head_dim
    q = self.idx_wq_b.forward(q_res, params).view(1, seq, hi, di).contiguous()
    _ext_rope(q[..., -rd:], self.inv_freq_compress, position=pos0)
    q = q[0]
    if self.ref_quant:
        q = fp4_qdq(q, 32, scale_e4m3=False)
    weights = self.idx_weights.forward(x, params).float()[0] * (di ** -0.5 * hi ** -0.5)
    k = min(self.index_topk, ec)
    indices = torch.full((seq, triton.cdiv(k, 32) * 32), -1, dtype=torch.int32, device=x.device)
    candidate = params.get(('v41_cand', inst)) if self.candidate_role == 'consumer' else None
    if ec <= self.candidate_topk_blocks * self.candidate_block_size:
        # Every visible block is retained; the causal mask already excludes all
        # remaining entries. Avoid specializing a kernel for each small block count.
        candidate = None
    if self.candidate_role == 'source':
        candidates = torch.empty((seq, min(self.candidate_topk_blocks,
            triton.cdiv(ec, self.candidate_block_size))), dtype=torch.int32, device=x.device)
    # Scores <= 32 MiB even at the checkpoint's full 1M context.
    tile = max(1, min(32, (32 * 2**20) // (4 * ec)))
    for start in range(0, seq, tile):
        count = min(tile, seq - start)
        scores = score_rows(q, keys, weights, ec, pos0, self.compress_rate,
                            start, count, candidate)
        if self.candidate_role == 'source':
            lens = (torch.arange(pos0 + start, pos0 + start + count, device=x.device) + 1) // self.compress_rate
            candidates[start:start + count] = candidate_blocks(scores, lens, ec,
                self.candidate_block_size, self.candidate_topk_blocks)
        top = scores.topk(k, dim=-1, sorted=False)
        idx = torch.where(top.values > float('-inf'), top.indices, ec)
        idx = idx.sort(dim=-1).values
        indices[start:start + count, :k] = torch.where(idx < ec, idx, -1).to(torch.int32)
    if self.candidate_role == 'source':
        params[('v41_cand', inst)] = CandidateBlocks(candidates, self.candidate_block_size)
    if (getattr(self, '_recipe_verify_pending', False) and seq == 1
            and 'prefill' not in params):
        _verify_actual_indexer(self, x, params, q_res, keys, ec, pos0, seq, inst, indices, k)
    return indices, k


def _verify_actual_indexer(module, x, params, q_res, keys, ec, pos0, seq, inst, actual, k):
    """One eager-reference comparison on real model activations before first output.

    The native function only projects/scores index queries; it does not advance
    recurrent state or write KV pools. Its candidate output uses a private dict.
    """
    from exllamav3.modules.dsv41 import DSV41Attention
    reference_params = dict(params)
    candidates = params.get(('v41_cand', inst))
    if isinstance(candidates, CandidateBlocks):
        reference_params[('v41_cand', inst)] = candidates.dense(ec)
    if isinstance(keys, PagedKeys):
        entries = torch.arange(ec, device=x.device)
        physical = keys.table[0, entries // keys.entries_per_page].long() * keys.entries_per_page
        physical += entries % keys.entries_per_page
        reference_keys = keys.pool.view(-1, 128).index_select(0, physical)
    else:
        reference_keys = keys
    expected, reference_k = DSV41Attention._v41_indexer(
        module, x, reference_params, q_res, reference_keys, ec, pos0, seq, inst)
    mismatches = (actual != expected).nonzero(as_tuple=False)
    receipt = dict(layer=module.layer_idx, position=pos0, entries=ec,
        role=module.candidate_role, k=k, indices_equal=torch.equal(actual, expected),
        differing_positions=int(mismatches.shape[0]),
        differences=[dict(row=int(row), column=int(col), actual=int(actual[row, col]),
                          expected=int(expected[row, col])) for row, col in mismatches[:10].tolist()])
    if module.candidate_role == 'source':
        receipt['candidate_mask_equal'] = torch.equal(candidates.dense(ec), reference_params[('v41_cand', inst)])
    module._recipe_indexer_receipts.append(receipt)
    if reference_k != k or not receipt['indices_equal'] or not receipt.get('candidate_mask_equal', True):
        raise RuntimeError(f'Actual-activation indexer parity failed: {receipt}')
    module._recipe_verify_pending = False


def configure(model, *, verify_first_decode=False):
    from exllamav3.modules.dsv41 import DSV41Attention
    count = 0
    receipts = []
    model._recipe_indexer_checks = receipts
    for module in model:
        if isinstance(module, DSV41Attention) and module.is_index_source:
            if (module.index_n_heads, module.index_head_dim) != (32, 128):
                raise ValueError('Only the pinned Flash indexer geometry is supported')
            module._gather_pool_idx = MethodType(_gather, module)
            module._v41_indexer = MethodType(_indexer, module)
            module._recipe_verify_pending = verify_first_decode
            module._recipe_indexer_receipts = receipts
            count += 1
    if count != 8:
        raise ValueError(f'Expected 8 DSV4.1 indexers, found {count}')
