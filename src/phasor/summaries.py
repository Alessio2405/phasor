"""Causal per-bucket residue summaries.

This is the component that turns the paper's "the softmax would filter it
anyway" from an assumption into a mechanism. Under a pure skip (Mixture-of-
Depths) an unrouted token is *absent* from the softmax, so it cannot be
down-weighted -- only skipped. Here it is present at coarse resolution.

**Causality is the easiest thing to get wrong here.** The summary a query at
position ``t`` sees must aggregate only keys ``j <= t``. We enforce it by
construction: summaries are exposed at block boundaries and the block used by
``t`` always ends at or before ``t``. :mod:`tests.test_causality` verifies it by
autograd rather than by inspection.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def summary_block_index(seq_len: int, stride: int, device=None) -> torch.Tensor:
    """For each position ``t``, the index of the last summary block ending ``<= t``.

    ``-1`` means "no summary available yet". With ``stride == 1`` this is just
    ``t`` (the summary is the exact running mean over ``0..t``).
    """
    t = torch.arange(seq_len, device=device)
    return ((t + 1) // stride) - 1


def causal_bucket_summaries(
    x: torch.Tensor,
    bucket: torch.Tensor,
    n_buckets: int,
    stride: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Running per-bucket means, exposed at block boundaries.

    Args:
        x: ``(Bsz, H, T, D)`` keys or values.
        bucket: ``(Bsz, H, T)`` long, the bucket of each *key*.
        n_buckets: ``B``.
        stride: block size. ``1`` gives exact running means at ``O(T*B*D)``
            memory; larger strides trade exactness for memory and are the
            practical setting at long context.

    Returns:
        ``means``: ``(Bsz, H, n_blocks, B, D)``
        ``counts``: ``(Bsz, H, n_blocks, B)``  -- zero means "bucket still empty",
        and the caller must mask those entries out of the softmax.
    """
    Bsz, H, T, D = x.shape
    if bucket.shape != (Bsz, H, T):
        raise ValueError(f"bucket shape {tuple(bucket.shape)} != {(Bsz, H, T)}")

    pad = (-T) % stride
    n_blocks = (T + pad) // stride

    onehot = F.one_hot(bucket, n_buckets).to(x.dtype)  # (Bsz, H, T, B)
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
        onehot = F.pad(onehot, (0, 0, 0, pad))

    x_b = x.view(Bsz, H, n_blocks, stride, D)
    o_b = onehot.view(Bsz, H, n_blocks, stride, n_buckets)

    block_sums = torch.einsum("bhnsd,bhnsk->bhnkd", x_b, o_b)  # (Bsz,H,n_blocks,B,D)
    block_counts = o_b.sum(dim=3)  # (Bsz,H,n_blocks,B)

    sums = block_sums.cumsum(dim=2)
    counts = block_counts.cumsum(dim=2)
    means = sums / counts.clamp(min=1.0).unsqueeze(-1)
    return means, counts


def gather_summaries_for_queries(
    means: torch.Tensor,
    counts: torch.Tensor,
    stride: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand block-level summaries to one per query position.

    Returns ``(Bsz, H, T, B, D)`` means and ``(Bsz, H, T, B)`` counts, where
    position ``t`` receives the last block ending at or before ``t``. Positions
    with no available block get zero counts (and are masked downstream).
    """
    Bsz, H, n_blocks, B, D = means.shape
    idx = summary_block_index(seq_len, stride, device=means.device)  # (T,)
    valid = idx >= 0
    safe = idx.clamp(min=0)

    m = means.index_select(2, safe)  # (Bsz,H,T,B,D)
    c = counts.index_select(2, safe)  # (Bsz,H,T,B)
    mask = valid.view(1, 1, seq_len, 1)
    c = c * mask
    m = m * mask.unsqueeze(-1)
    return m, c


class IncrementalSummaries:
    """Decoding-time state: ``O(B * D)`` per head, independent of sequence length.

    Kept deliberately separate from the training path so that
    :mod:`tests.test_summaries` can assert the two agree. If they ever diverge,
    the model trains on a different function from the one it decodes with.
    """

    def __init__(self, n_buckets: int, d_k: int, d_v: int, device=None, dtype=torch.float32):
        self.n_buckets = n_buckets
        self.k_sum = torch.zeros(n_buckets, d_k, device=device, dtype=dtype)
        self.v_sum = torch.zeros(n_buckets, d_v, device=device, dtype=dtype)
        self.counts = torch.zeros(n_buckets, device=device, dtype=dtype)

    def update(self, bucket: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.k_sum[bucket] += k
        self.v_sum[bucket] += v
        self.counts[bucket] += 1

    def read(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        denom = self.counts.clamp(min=1.0).unsqueeze(-1)
        return self.k_sum / denom, self.v_sum / denom, self.counts
