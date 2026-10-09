"""Ablation variants, all built from the same :class:`PhasorAttention` class.

The point of routing every baseline through one class is that A1, A3 and A4
become genuine single-variable interventions: the projections, the RoPE split,
the scaling and the softmax are literally the same code. A separately written
"Reformer baseline" would differ in a dozen incidental ways and the ablation
would measure those instead.

What this module does **not** provide: faithful reimplementations of Routing
Transformer, NSA or MoBA. Those are external systems and the protocol
(Section 5.3) says to either reimplement them properly or state that they are
missing -- not to fake them with a near-miss.
"""

from __future__ import annotations

from .attention import PhasorAttention

#: name -> kwargs overrides. Consumed by :func:`make_attention`.
VARIANTS: dict[str, dict] = {
    # -- main arms ---------------------------------------------------------
    "phasor": dict(
        pattern="phasor", rotation="cayley", route_on="content", use_summaries=True
    ),
    "dense": dict(pattern="dense", rope_fraction=1.0),
    "dense_partial_rope": dict(pattern="dense"),
    # -- A1: is the learned frame doing anything? --------------------------
    # Negative arm = Reformer-style random projection LSH. If this ties with
    # "phasor", the central claim fails (the fixed-hash null, Roller et al. 2021).
    "a1_frozen_rotation": dict(
        pattern="phasor", rotation="frozen_random", route_on="content", use_summaries=True
    ),
    "a1_householder": dict(
        pattern="phasor", rotation="householder", route_on="content", use_summaries=True
    ),
    # -- A3: the RoPE-decoupling claim (Proposition 2) ---------------------
    "a3_route_on_rope": dict(
        pattern="phasor", rotation="cayley", route_on="rope", use_summaries=True
    ),
    "a3_route_on_full": dict(
        pattern="phasor", rotation="cayley", route_on="full", use_summaries=True
    ),
    # -- A4: the attention-visible fallback --------------------------------
    # Negative arm = pure skip. Excluded keys are absent from the softmax, so
    # attention cannot down-weight them; only the residual stream carries them.
    "a4_pure_skip": dict(
        pattern="phasor", rotation="cayley", route_on="content", use_summaries=False
    ),
    "a4_summaries_full_space": dict(
        pattern="phasor", route_on="content", use_summaries=True, summary_space="full"
    ),
    # -- A6 ----------------------------------------------------------------
    "a6_no_balance": dict(pattern="phasor", route_on="content", balance_lambda=0.0),
}


def make_attention(variant: str, d_model: int, n_heads: int, **kwargs) -> PhasorAttention:
    """Build an attention module by ablation name.

    Explicit ``kwargs`` win over the variant's defaults, so a sweep can override
    ``window`` or ``rope_fraction`` without editing :data:`VARIANTS`.
    """
    if variant not in VARIANTS:
        raise KeyError(f"unknown variant {variant!r}; known: {sorted(VARIANTS)}")
    cfg = dict(VARIANTS[variant])
    cfg.update(kwargs)
    return PhasorAttention(d_model=d_model, n_heads=n_heads, **cfg)
