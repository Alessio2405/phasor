"""The PHASOR attention layer (Algorithm 1 of the paper).

**This is a correctness reference, not a fast kernel.** It materialises the full
``(T, T)`` score matrix and applies the routing decision as a mask, so it costs
``O(T^2)`` in time and memory -- the same as dense attention. The ``O(T^1.5)``
claim in Section 4.1 of the paper depends on a bucket-sorted, chunked gather
path (Reformer-style) which is deliberately *not* implemented here; see
``NOTES_FOR_PAPER.md``. Use this module to validate semantics and to measure
pattern quality (recall, load entropy, summary mass), not to measure speed.

Every baseline in :mod:`phasor.baselines` is this same class with one flag
changed, so ablations A1/A3/A4 are genuine single-variable interventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_partial_rope, build_rope_cache
from .router import PhasorRouter, load_balance_entropy_loss, normalised_load_entropy
from .summaries import causal_bucket_summaries, gather_summaries_for_queries


@dataclass
class AttentionAux:
    """Diagnostics. Everything the protocol in Section 5.4 asks for."""

    balance_loss: torch.Tensor | None = None
    load_entropy: float | None = None
    summary_mass: float | None = None
    pattern_density: float | None = None
    bucket_q: torch.Tensor | None = None
    bucket_k: torch.Tensor | None = None
    extras: dict = field(default_factory=dict)


class PhasorAttention(nn.Module):
    """Causal attention with phase-aware orthogonal routing.

    Args:
        d_model: model width.
        n_heads: number of heads.
        rope_fraction: ``rho = d_p / d_head``. ``1.0`` is full RoPE (and, by
            Proposition 2, destroys content routing -- that is ablation A3's
            negative arm). ``0.0`` is NoPE.
        route_on: ``"content"`` routes on the RoPE-free dims (ours);
            ``"rope"`` routes on the rotated dims (the A3 negative arm);
            ``"full"`` routes on the whole head.
        pattern: ``"phasor"`` or ``"dense"``. ``"dense"`` bypasses routing
            entirely and is the dense baseline with identical projections.
        rotation: ``"cayley"`` / ``"householder"`` / ``"frozen_random"``.
            ``"frozen_random"`` is ablation A1's negative arm (Reformer LSH).
        use_summaries: if False, unrouted keys are simply absent from the
            softmax -- the Mixture-of-Depths-style pure skip (ablation A4).
        summary_bias_init: initial logit bias on the residue branch. Negative
            keeps the branch quiet at init; see NOTES_FOR_PAPER.md (F3).
        summary_space: ``"content"`` pools the RoPE-free part of keys;
            ``"full"`` pools post-RoPE keys, which suffers phase cancellation
            (see NOTES_FOR_PAPER.md). Default ``"content"``.
        window: sliding-window width; keys within ``window`` of the query are
            always visible regardless of routing. Must be ``>= 1`` so that a
            query can always see itself.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rope_fraction: float = 0.5,
        route_on: str = "content",
        pattern: str = "phasor",
        rotation: str = "cayley",
        n_rounds: int = 1,
        window: int = 64,
        use_summaries: bool = True,
        summary_space: str = "content",
        summary_stride: int = 1,
        summary_bias_init: float = -4.0,
        gate: bool = True,
        balance_lambda: float = 1e-2,
        router_temperature: float = 1.0,
        straight_through: bool = False,
        rope_base: float = 10000.0,
        dropout: float = 0.0,
        bias: bool = False,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if window < 1:
            raise ValueError("window must be >= 1 so a query can attend to itself")
        if pattern not in ("phasor", "dense"):
            raise ValueError(f"unknown pattern {pattern!r}")
        if route_on not in ("content", "rope", "full"):
            raise ValueError(f"unknown route_on {route_on!r}")
        if summary_space not in ("content", "full"):
            raise ValueError(f"unknown summary_space {summary_space!r}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.pattern = pattern
        self.route_on = route_on
        self.window = window
        self.use_summaries = use_summaries and pattern == "phasor"
        self.summary_space = summary_space
        self.summary_stride = summary_stride
        self.gate = gate
        self.balance_lambda = balance_lambda
        self.dropout = dropout

        # Positional split. d_pos must be even for RoPE's planar rotations.
        d_pos = int(round(rope_fraction * self.d_head))
        d_pos -= d_pos % 2
        d_pos = max(0, min(self.d_head, d_pos))
        self.d_pos = d_pos
        self.d_c = self.d_head - d_pos

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        if pattern == "phasor":
            d_route = {"content": self.d_c, "rope": self.d_pos, "full": self.d_head}[route_on]
            if d_route == 0:
                raise ValueError(
                    f"route_on={route_on!r} gives a zero-width routing subspace "
                    f"(d_pos={self.d_pos}, d_c={self.d_c}); adjust rope_fraction"
                )
            self.d_route = d_route
            self.router = PhasorRouter(
                d_route,
                rotation=rotation,
                n_rounds=n_rounds,
                temperature=router_temperature,
                straight_through=straight_through,
            )
            self.n_buckets = self.router.n_buckets
            if self.use_summaries:
                # The residue branch scores over a different subspace from the
                # fine branch, so it needs its own calibration inside the shared
                # softmax. Allocated only when used, so that ablation A4's two
                # arms differ by exactly these parameters and nothing else.
                self.summary_log_scale = nn.Parameter(torch.zeros(n_heads))
                # Initialised negative so the residue branch starts quiet and
                # the model has to learn to use it. With a zero bias it takes
                # ~50% of the attention mass at init (see NOTES_FOR_PAPER.md,
                # Finding F3), far outside the 2-25% band prediction P3 assumes.
                self.summary_bias = nn.Parameter(torch.full((n_heads,), summary_bias_init))
        else:
            self.router = None
            self.n_buckets = 0

        cos, sin = build_rope_cache(max_seq_len, self.d_pos, base=rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    # -- helpers ------------------------------------------------------------

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        Bsz, T, _ = x.shape
        return x.view(Bsz, T, self.n_heads, self.d_head).transpose(1, 2)

    def _routing_input(self, full: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        if self.route_on == "content":
            return content
        if self.route_on == "rope":
            return full[..., : self.d_pos]
        return full

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, AttentionAux]:
        Bsz, T, _ = x.shape
        device = x.device

        if T > self.rope_cos.shape[0]:
            cos, sin = build_rope_cache(T, self.d_pos, device=device, dtype=x.dtype)
        else:
            cos, sin = self.rope_cos.to(device), self.rope_sin.to(device)

        q = self._heads(self.q_proj(x))
        k = self._heads(self.k_proj(x))
        v = self._heads(self.v_proj(x))

        q_full, q_c = apply_partial_rope(q, self.d_pos, cos, sin)
        k_full, k_c = apply_partial_rope(k, self.d_pos, cos, sin)

        scale = self.d_head ** -0.5
        logits = (q_full @ k_full.transpose(-1, -2)) * scale  # (Bsz,H,T,T)

        causal = torch.ones(T, T, dtype=torch.bool, device=device).tril()
        aux = AttentionAux()

        summary_logits = None
        vbar = None
        gate = None

        if self.pattern == "dense":
            allowed = causal.view(1, 1, T, T).expand(Bsz, self.n_heads, T, T)
        else:
            r_q = self.router(self._routing_input(q_full, q_c))
            r_k = self.router(self._routing_input(k_full, k_c))
            bq, bk = r_q.bucket, r_k.bucket  # (Bsz,H,T,R)

            same = (bq.unsqueeze(-2) == bk.unsqueeze(-3)).any(dim=-1)  # (Bsz,H,T,T)
            offs = torch.arange(T, device=device)
            near = (offs.view(-1, 1) - offs.view(1, -1)).abs() < self.window
            allowed = causal.view(1, 1, T, T) & (same | near.view(1, 1, T, T))

            gate = r_q.gate.mean(dim=-1) if self.gate else None  # (Bsz,H,T)

            if self.balance_lambda > 0:
                aux.balance_loss = self.balance_lambda * load_balance_entropy_loss(
                    r_k.scores, bk, self.router.n_buckets, self.router.temperature
                )
            aux.load_entropy = normalised_load_entropy(bk, self.router.n_buckets)
            aux.bucket_q, aux.bucket_k = bq, bk

            if self.use_summaries:
                k_pool = k_c if self.summary_space == "content" else k_full
                # Values are always pooled whole: they carry no positional phase.
                means_k, counts = causal_bucket_summaries(
                    k_pool, bk[..., 0], self.router.n_buckets, self.summary_stride
                )
                means_v, _ = causal_bucket_summaries(
                    v, bk[..., 0], self.router.n_buckets, self.summary_stride
                )
                kbar, cnt = gather_summaries_for_queries(means_k, counts, self.summary_stride, T)
                vbar, _ = gather_summaries_for_queries(means_v, counts, self.summary_stride, T)

                q_pool = q_c if self.summary_space == "content" else q_full
                s_logits = torch.einsum("bhtd,bhtkd->bhtk", q_pool, kbar) * scale
                s_scale = self.summary_log_scale.exp().view(1, -1, 1, 1)
                s_bias = self.summary_bias.view(1, -1, 1, 1)
                s_logits = s_logits * s_scale + s_bias

                # Mask: buckets the query already attends to finely, and buckets
                # that are still empty at this position.
                selected = F.one_hot(bq, self.router.n_buckets).sum(dim=-2) > 0  # (Bsz,H,T,B)
                s_mask = (cnt > 0) & (~selected)
                summary_logits = s_logits.masked_fill(~s_mask, float("-inf"))
                aux.extras["summary_valid_frac"] = float(s_mask.float().mean().detach())

        logits = logits.masked_fill(~allowed, float("-inf"))
        # Normalised against the *causal* cells, not the full square, so that
        # dense attention reports 1.0 and the number is comparable across
        # sequence lengths. This must match :func:`phasor.metrics.pattern_density`.
        causal_cells = T * (T + 1) / 2
        aux.pattern_density = float(
            (allowed.float().sum() / (Bsz * self.n_heads * causal_cells)).detach()
        )

        if summary_logits is not None:
            joint = torch.cat([logits, summary_logits], dim=-1)
            probs = F.softmax(joint, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)
            p_fine, p_sum = probs[..., :T], probs[..., T:]
            aux.summary_mass = float(p_sum.sum(dim=-1).mean().detach())
            if self.dropout > 0 and self.training:
                p_fine = F.dropout(p_fine, self.dropout)
                p_sum = F.dropout(p_sum, self.dropout)
            out = p_fine @ v + torch.einsum("bhtk,bhtkd->bhtd", p_sum, vbar)
        else:
            probs = torch.nan_to_num(F.softmax(logits, dim=-1), nan=0.0)
            if self.dropout > 0 and self.training:
                probs = F.dropout(probs, self.dropout)
            out = probs @ v
            aux.summary_mass = 0.0

        if gate is not None:
            out = out * gate.unsqueeze(-1)

        out = out.transpose(1, 2).reshape(Bsz, T, self.d_model)
        out = self.o_proj(out)
        return (out, aux) if return_aux else out
