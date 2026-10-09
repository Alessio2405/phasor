"""Numerical verification of Proposition 2 (phase contamination).

These tests do not prove the proposition -- the paper does that -- but they check
that the *implementation* exhibits the predicted behaviour, and they are the
in-repo statement of what experiment E0 is looking for at the scale of a real
model.

The prediction: a router reading RoPE-free dimensions is *exactly*
position-invariant, while one reading RoPE-rotated dimensions loses that
invariance as rotary phase accumulates.

Running these revealed that the paper's asymptotic ``1/B`` overstates the
aggregate effect on a real geometric RoPE spectrum -- see Finding F1 in
``NOTES_FOR_PAPER.md``. The tests assert what is actually observed, not what the
paper's corollary currently says.
"""

import pytest
import torch

from phasor.rope import apply_rope, build_rope_cache
from phasor.router import PhasorRouter


def _agreement_with_first(buckets: torch.Tensor) -> float:
    return float((buckets == buckets[0]).float().mean())


def test_content_routing_is_exactly_position_invariant():
    """The whole point of the decoupling: bucket depends on content alone."""
    d_c, T = 16, 512
    router = PhasorRouter(d_c=d_c)
    u = torch.randn(d_c)
    repeated = u.unsqueeze(0).expand(T, d_c)  # same content at every position

    buckets = router(repeated).bucket.squeeze(-1)
    assert _agreement_with_first(buckets) == 1.0


def test_rope_routing_destroys_content_invariance():
    """Proposition 2's practical content: identical content stops co-routing."""
    d, T = 16, 2048
    router = PhasorRouter(d_c=d)
    cos, sin = build_rope_cache(T, d)

    torch.manual_seed(1)
    agreements = []
    for _ in range(24):  # average over content directions
        u = torch.randn(d)
        repeated = u.unsqueeze(0).expand(T, d)
        rotated = apply_rope(repeated, cos, sin)
        agreements.append(_agreement_with_first(router(rotated).bucket.squeeze(-1)))

    mean_agreement = sum(agreements) / len(agreements)
    assert mean_agreement < 0.35, (
        f"agreement {mean_agreement:.4f} is high; if this stays near 1.0, "
        "Proposition 2 has no practical force and the RoPE-decoupling "
        "contribution should be withdrawn"
    )


def test_rope_routing_collapses_the_effective_bucket_count():
    """Finding F1 (see NOTES_FOR_PAPER.md): the paper overstates the limit.

    Proposition 2's asymptotic ``1/B`` assumes every routing plane
    equidistributes. RoPE's frequency spectrum is geometric, so the
    low-frequency planes barely move over any realistic context and the argmax
    stays confined to a handful of axes. Empirically the agreement plateaus near
    0.16 with ~5 of 32 buckets ever visited -- far above ``1/B = 0.031``.

    That is not a reprieve: confining routing to a fifth of the codebook *is*
    load collapse, and it is arguably a worse failure than uniform hashing. But
    it is a different failure from the one the paper's corollary describes, and
    the paper should say so.
    """
    d, T = 16, 4096
    router = PhasorRouter(d_c=d)
    B = router.n_buckets
    cos, sin = build_rope_cache(T, d)

    torch.manual_seed(1)
    used = []
    for _ in range(16):
        u = torch.randn(d)
        b = router(apply_rope(u.unsqueeze(0).expand(T, d), cos, sin)).bucket.squeeze(-1)
        used.append(len(b.unique()))
    mean_used = sum(used) / len(used)

    assert mean_used < B / 3, (
        f"expected routing to be confined to a small subset of buckets, "
        f"got {mean_used:.1f} of {B}"
    )
    assert mean_used > 1, "routing collapsed to a single bucket; check the setup"


def test_contamination_grows_with_context_length():
    """Prediction P1: the effect must *widen* with context, not be constant.

    A flat curve would mean the mechanism is something other than accumulated
    rotary phase, and P1 would have failed.
    """
    d = 16
    router = PhasorRouter(d_c=d)
    torch.manual_seed(2)
    u = torch.randn(64, d)

    agreements = {}
    for T in (32, 256, 4096):
        cos, sin = build_rope_cache(T, d)
        per_vec = []
        for i in range(u.shape[0]):
            rep = u[i].unsqueeze(0).expand(T, d)
            b = router(apply_rope(rep, cos, sin)).bucket.squeeze(-1)
            per_vec.append(_agreement_with_first(b))
        agreements[T] = sum(per_vec) / len(per_vec)

    assert agreements[32] > agreements[256] > agreements[4096], (
        f"agreement should fall monotonically with context, got {agreements}"
    )


def test_high_frequency_planes_contaminate_faster():
    """Proposition 2's remainder term: contamination scales with ``theta * W``.

    A smaller RoPE base means higher frequencies means faster decorrelation. Note
    that the relationship is only reliable at the *extremes* of the base range;
    in between, which plane wins the argmax varies and the curve is not
    monotone (Finding F1).
    """
    d, T = 16, 2048
    router = PhasorRouter(d_c=d)
    torch.manual_seed(3)
    u = torch.randn(48, d)

    def mean_agreement(base):
        cos, sin = build_rope_cache(T, d, base=base)
        vals = []
        for i in range(u.shape[0]):
            rep = u[i].unsqueeze(0).expand(T, d)
            vals.append(_agreement_with_first(router(apply_rope(rep, cos, sin)).bucket.squeeze(-1)))
        return sum(vals) / len(vals)

    high_freq = mean_agreement(base=100.0)
    low_freq = mean_agreement(base=10_000_000.0)
    assert high_freq < low_freq, (
        f"high-frequency RoPE should scramble routing more, got "
        f"high={high_freq:.4f} low={low_freq:.4f}"
    )


@pytest.mark.parametrize("route_on,expect_invariant", [("content", True), ("rope", False)])
def test_attention_layer_inherits_the_property(route_on, expect_invariant):
    """The same property, measured through the full attention module."""
    from phasor.attention import PhasorAttention
    from phasor.rope import apply_partial_rope, build_rope_cache as brc

    # A realistic head width matters here. At d_head = 16 there are only four
    # RoPE planes and contamination is mild (~0.6 agreement); at d_head = 64 it
    # drops to ~0.2. Head width is part of the effect size, which the paper's
    # Proposition 2 does not currently say.
    attn = PhasorAttention(
        d_model=128, n_heads=2, rope_fraction=0.5, route_on=route_on, window=1
    )
    T = 2048
    # One token embedding repeated: identical content at every position.
    x = torch.randn(1, 1, 128).expand(1, T, 128).contiguous()

    cos, sin = brc(T, attn.d_pos, dtype=x.dtype)
    k = attn._heads(attn.k_proj(x))
    k_full, k_c = apply_partial_rope(k, attn.d_pos, cos, sin)
    buckets = attn.router(attn._routing_input(k_full, k_c)).bucket  # (1,H,T,R)

    per_head = [(buckets[0, h, :, 0] == buckets[0, h, 0, 0]).float().mean() for h in range(2)]
    agreement = float(sum(per_head) / len(per_head))

    if expect_invariant:
        assert agreement == 1.0
    else:
        assert agreement < 0.4
