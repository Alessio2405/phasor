"""Diagnostics from Section 5.4 of the paper.

The single most informative number for a sparse pattern is
:func:`topk_mass_recall`: of the attention mass dense attention puts on its top-k
keys, how much survives the sparse pattern. Perplexity tells you whether the
model coped; recall tells you whether the *pattern* was any good, which is what
the router is actually responsible for.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def normalised_entropy(counts: torch.Tensor) -> float:
    """``H(pi) / log n`` in ``[0, 1]`` for a non-negative count vector."""
    counts = counts.float().reshape(-1)
    total = counts.sum()
    if total <= 0:
        return 0.0
    pi = counts / total
    h = -(pi * (pi + 1e-12).log()).sum()
    return float(h / math.log(len(pi)))


@torch.no_grad()
def dense_reference_probs(attn, x: torch.Tensor) -> torch.Tensor:
    """Dense causal attention probabilities using ``attn``'s own projections.

    Reusing the module's weights matters: we want to know what *this* model would
    attend to, not what some other model would.
    """
    from .rope import apply_partial_rope, build_rope_cache

    Bsz, T, _ = x.shape
    cos, sin = build_rope_cache(T, attn.d_pos, device=x.device, dtype=x.dtype)
    q = attn._heads(attn.q_proj(x))
    k = attn._heads(attn.k_proj(x))
    q_full, _ = apply_partial_rope(q, attn.d_pos, cos, sin)
    k_full, _ = apply_partial_rope(k, attn.d_pos, cos, sin)
    logits = (q_full @ k_full.transpose(-1, -2)) * attn.d_head**-0.5
    causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
    return F.softmax(logits.masked_fill(~causal, float("-inf")), dim=-1)


@torch.no_grad()
def allowed_mask(attn, x: torch.Tensor) -> torch.Tensor:
    """The boolean ``(Bsz, H, T, T)`` pattern the router admits."""
    from .rope import apply_partial_rope, build_rope_cache

    Bsz, T, _ = x.shape
    if attn.pattern == "dense":
        return torch.ones(Bsz, attn.n_heads, T, T, dtype=torch.bool, device=x.device).tril()

    cos, sin = build_rope_cache(T, attn.d_pos, device=x.device, dtype=x.dtype)
    q = attn._heads(attn.q_proj(x))
    k = attn._heads(attn.k_proj(x))
    q_full, q_c = apply_partial_rope(q, attn.d_pos, cos, sin)
    k_full, k_c = apply_partial_rope(k, attn.d_pos, cos, sin)
    bq = attn.router(attn._routing_input(q_full, q_c)).bucket
    bk = attn.router(attn._routing_input(k_full, k_c)).bucket
    same = (bq.unsqueeze(-2) == bk.unsqueeze(-3)).any(dim=-1)
    offs = torch.arange(T, device=x.device)
    near = (offs.view(-1, 1) - offs.view(1, -1)).abs() < attn.window
    causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
    return causal.view(1, 1, T, T) & (same | near.view(1, 1, T, T))


@torch.no_grad()
def topk_mass_recall(attn, x: torch.Tensor, k: int = 32) -> float:
    """Fraction of dense top-``k`` attention mass that the sparse pattern retains.

    1.0 means the pattern never excludes anything dense attention cared about.
    """
    probs = dense_reference_probs(attn, x)
    mask = allowed_mask(attn, x)
    T = probs.shape[-1]
    kk = min(k, T)
    top_val, top_idx = probs.topk(kk, dim=-1)
    kept = mask.gather(-1, top_idx).to(top_val.dtype)
    num = (top_val * kept).sum()
    den = top_val.sum().clamp(min=1e-12)
    return float(num / den)


@torch.no_grad()
def pattern_density(attn, x: torch.Tensor) -> float:
    """Fraction of the causal ``(T, T)`` lower triangle the pattern admits.

    This is the honest stand-in for FLOP savings in the reference
    implementation, which does not actually skip the masked work.
    """
    mask = allowed_mask(attn, x)
    T = mask.shape[-1]
    causal_cells = T * (T + 1) / 2
    return float(mask.float().sum() / (mask.shape[0] * mask.shape[1] * causal_cells))


@torch.no_grad()
def position_bucket_mutual_information(
    bucket: torch.Tensor, n_buckets: int, n_position_bins: int = 16
) -> float:
    """``I(bucket; position_bin) / H(position_bin)``, in ``[0, 1]``.

    This is the core estimator for experiment E0. A high value means the router
    is hashing by *position* rather than by content -- the failure Proposition 2
    predicts for routers that read RoPE-rotated dimensions.
    """
    flat_bucket = bucket.reshape(-1)
    T = bucket.shape[-1] if bucket.dim() >= 2 else bucket.shape[0]
    n_rep = flat_bucket.numel() // T
    pos = torch.arange(T, device=bucket.device)
    pos_bin = (pos.float() / T * n_position_bins).long().clamp(max=n_position_bins - 1)
    flat_pos = pos_bin.repeat(n_rep)

    joint = torch.zeros(n_buckets, n_position_bins, device=bucket.device)
    idx = flat_bucket * n_position_bins + flat_pos
    joint.view(-1).scatter_add_(0, idx, torch.ones_like(idx, dtype=joint.dtype))
    joint = joint / joint.sum().clamp(min=1.0)

    p_b = joint.sum(dim=1, keepdim=True)
    p_p = joint.sum(dim=0, keepdim=True)
    nz = joint > 0
    ratio = joint[nz] / (p_b.expand_as(joint)[nz] * p_p.expand_as(joint)[nz])
    mi = (joint[nz] * torch.log(ratio)).sum()
    h_pos = -(p_p[p_p > 0] * p_p[p_p > 0].log()).sum()
    return float(mi / h_pos.clamp(min=1e-12))
