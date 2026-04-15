"""Triton-accelerated kernels for mirror descent + belief propagation.

Contains:
1. Fused scatter-logsumexp Triton kernel (used by BeliefPropagation.forward).
"""

from __future__ import annotations

import torch

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: fused scatter logsumexp
# ---------------------------------------------------------------------------


@triton.jit
def _scatter_logsumexp_kernel(
    proc_ptr,
    gather_idx_ptr,
    group_offsets_ptr,
    out_ptr,
    rest_dom: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """Segmented logsumexp: for each target row b, compute
    logsumexp over the gathered source rows that map to b.

    Grid: (b_idx_dom, cdiv(rest_dom, BLOCK_J))
    """
    pid_b = tl.program_id(0)
    pid_j = tl.program_id(1)

    start = tl.load(group_offsets_ptr + pid_b)
    end = tl.load(group_offsets_ptr + pid_b + 1)

    j_base = pid_j * BLOCK_J
    j_offsets = j_base + tl.arange(0, BLOCK_J)
    j_mask = j_offsets < rest_dom

    # Pass 1: find max over segment
    max_val = tl.full([BLOCK_J], float("-inf"), dtype=tl.float32)
    for k in range(start, end):
        src_row = tl.load(gather_idx_ptr + k)
        vals = tl.load(
            proc_ptr + src_row * rest_dom + j_offsets,
            mask=j_mask,
            other=float("-inf"),
        )
        max_val = tl.maximum(max_val, vals)

    # Pass 2: sum of exp(x - max)
    acc = tl.zeros([BLOCK_J], dtype=tl.float32)
    for k in range(start, end):
        src_row = tl.load(gather_idx_ptr + k)
        vals = tl.load(
            proc_ptr + src_row * rest_dom + j_offsets,
            mask=j_mask,
            other=float("-inf"),
        )
        acc += tl.exp(vals - max_val)

    result = tl.log(acc) + max_val
    tl.store(out_ptr + pid_b * rest_dom + j_offsets, result, mask=j_mask)


def scatter_logsumexp(
    proc: torch.Tensor,
    group_offsets: torch.Tensor,
    b_idx_dom: int,
    rest_dom: int,
) -> torch.Tensor:
    """Fused scatter logsumexp using Triton.

    Expects proc rows already gathered and sorted by target group
    (i.e. after index_select with gather_idx).

    Args:
        proc: (n_pairs, rest_dom) float32, gathered source rows in log-space.
        group_offsets: (b_idx_dom + 1,) int64, CSR offsets into proc rows.
        b_idx_dom: number of target rows.
        rest_dom: number of columns.

    Returns:
        (b_idx_dom, rest_dom) float32 result in log-space.
    """
    proc = proc.contiguous()
    out = proc.new_full((b_idx_dom, rest_dom), float("-inf"))

    if b_idx_dom == 0 or rest_dom == 0:
        return out

    n_rows = proc.shape[0]
    identity = torch.arange(n_rows, device=proc.device, dtype=torch.int64)
    BLOCK_J = min(triton.next_power_of_2(rest_dom), 1024)
    grid = (b_idx_dom, triton.cdiv(rest_dom, BLOCK_J))
    _scatter_logsumexp_kernel[grid](
        proc, identity, group_offsets, out,
        rest_dom=rest_dom, BLOCK_J=BLOCK_J,
    )
    return out
