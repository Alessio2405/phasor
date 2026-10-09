"""PHASOR: Phase-Aware Sparse Attention with Learned Orthogonal Routing.

Reference implementation. See ``paper/PHASOR_paper.pdf`` for the method and
``NOTES_FOR_PAPER.md`` for design decisions this code forced that the paper does
not yet cover.

No empirical claim in the paper is supported by results yet. This package exists
so that experiment E0 and ablations A1/A3/A4 can be run, not so that they can be
asserted.
"""

from .attention import AttentionAux, PhasorAttention
from .baselines import VARIANTS, make_attention
from .model import GPT, GPTConfig
from .rope import apply_partial_rope, apply_rope, build_rope_cache, rope_frequencies
from .rotation import (
    CayleyRotation,
    FrozenRandomRotation,
    HouseholderRotation,
    Rotation,
    build_rotation,
)
from .router import (
    PhasorRouter,
    RouterOutput,
    bucket_marginal,
    load_balance_entropy_loss,
    normalised_load_entropy,
)
from .summaries import (
    IncrementalSummaries,
    causal_bucket_summaries,
    gather_summaries_for_queries,
    summary_block_index,
)

__version__ = "0.1.0"

__all__ = [
    "PhasorAttention",
    "AttentionAux",
    "PhasorRouter",
    "RouterOutput",
    "Rotation",
    "CayleyRotation",
    "HouseholderRotation",
    "FrozenRandomRotation",
    "build_rotation",
    "build_rope_cache",
    "apply_rope",
    "apply_partial_rope",
    "rope_frequencies",
    "causal_bucket_summaries",
    "gather_summaries_for_queries",
    "summary_block_index",
    "IncrementalSummaries",
    "bucket_marginal",
    "load_balance_entropy_loss",
    "normalised_load_entropy",
    "make_attention",
    "VARIANTS",
    "GPT",
    "GPTConfig",
]
