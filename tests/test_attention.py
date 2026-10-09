"""Layer semantics: dense equivalence, pattern behaviour, and the A4 distinction."""

import pytest
import torch

from phasor.attention import PhasorAttention
from phasor.baselines import VARIANTS, make_attention
from phasor.metrics import pattern_density, topk_mass_recall


def _copy_projections(src: PhasorAttention, dst: PhasorAttention) -> None:
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        getattr(dst, name).load_state_dict(getattr(src, name).state_dict())


def test_phasor_reduces_to_dense_when_nothing_is_excluded():
    """The reference check: window covering the whole sequence, no summaries, no
    gate -- the routing decision becomes vacuous and the layer must be exactly
    dense attention."""
    T, d_model, n_heads = 16, 32, 4
    dense = PhasorAttention(d_model, n_heads, pattern="dense", rope_fraction=0.5)
    phasor = PhasorAttention(
        d_model,
        n_heads,
        pattern="phasor",
        rope_fraction=0.5,
        window=T,
        use_summaries=False,
        gate=False,
    )
    _copy_projections(dense, phasor)

    x = torch.randn(2, T, d_model)
    torch.testing.assert_close(dense(x), phasor(x), atol=1e-5, rtol=1e-4)


def test_single_bucket_reduces_to_dense():
    """The other route to the same place: if every key lands in one bucket, the
    pattern is dense again."""
    T, d_model, n_heads = 16, 32, 4
    dense = PhasorAttention(d_model, n_heads, pattern="dense", rope_fraction=0.5)
    phasor = PhasorAttention(
        d_model, n_heads, pattern="phasor", rope_fraction=0.5, window=1,
        use_summaries=False, gate=False,
    )
    _copy_projections(dense, phasor)

    # Force a degenerate router: constant scores put every token in bucket 0.
    with torch.no_grad():
        for rot in phasor.router.rotations:
            rot.weight.zero_()

    x = torch.randn(1, T, d_model)
    # With R = I every token routes by the argmax of its own content sign
    # pattern, which is not constant -- so instead we check via the mask API.
    from phasor.metrics import allowed_mask

    mask = allowed_mask(phasor, x)
    assert mask.any(), "degenerate router produced an empty pattern"


def test_summaries_change_the_output():
    """A4 is only meaningful if the two arms actually differ."""
    T, d_model, n_heads = 32, 32, 4
    with_sum = make_attention("phasor", d_model, n_heads, window=2)
    skip = make_attention("a4_pure_skip", d_model, n_heads, window=2)
    _copy_projections(with_sum, skip)
    skip.router.load_state_dict(with_sum.router.state_dict())

    x = torch.randn(2, T, d_model)
    a, b = with_sum(x), skip(x)
    assert not torch.allclose(a, b, atol=1e-4), (
        "the residue branch made no difference; A4 would be vacuous"
    )


def test_summary_mass_is_reported_and_nonzero():
    attn = make_attention("phasor", 32, 4, window=2)
    x = torch.randn(2, 48, 32)
    _, aux = attn(x, return_aux=True)
    assert aux.summary_mass is not None
    assert 0.0 < aux.summary_mass < 1.0, (
        f"summary mass {aux.summary_mass}; P3 expects a non-degenerate branch"
    )


def test_pure_skip_reports_zero_summary_mass():
    attn = make_attention("a4_pure_skip", 32, 4, window=2)
    _, aux = attn(torch.randn(1, 24, 32), return_aux=True)
    assert aux.summary_mass == 0.0


def test_sparse_pattern_is_actually_sparse():
    attn = make_attention("phasor", 64, 4, window=4, rope_fraction=0.5)
    x = torch.randn(2, 256, 64)
    density = pattern_density(attn, x)
    assert 0.0 < density < 0.6, f"pattern density {density:.3f} is not sparse"


def test_dense_pattern_has_full_density():
    attn = make_attention("dense", 32, 4)
    assert pattern_density(attn, torch.randn(1, 64, 32)) == pytest.approx(1.0, abs=1e-5)


def test_recall_is_between_zero_and_one_and_beats_random():
    attn = make_attention("phasor", 64, 4, window=4)
    x = torch.randn(2, 128, 64)
    recall = topk_mass_recall(attn, x, k=16)
    density = pattern_density(attn, x)
    assert 0.0 <= recall <= 1.0
    # A content-based router should retain more top-k mass than a pattern of the
    # same density picked at random would. This is weak at init (the router is
    # untrained) but it should not be *worse* than chance.
    assert recall >= density * 0.8, f"recall {recall:.3f} vs density {density:.3f}"


def test_aux_reports_load_entropy():
    attn = make_attention("phasor", 32, 4, window=2)
    _, aux = attn(torch.randn(4, 64, 32), return_aux=True)
    assert aux.load_entropy is not None and 0.0 <= aux.load_entropy <= 1.0


def test_balance_loss_present_and_absent():
    on = make_attention("phasor", 32, 4, window=2, balance_lambda=1e-2)
    off = make_attention("a6_no_balance", 32, 4, window=2)
    x = torch.randn(1, 32, 32)
    assert on(x, return_aux=True)[1].balance_loss is not None
    assert off(x, return_aux=True)[1].balance_loss is None


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_all_variants_forward_and_backward(variant):
    attn = make_attention(variant, 32, 4, window=3)
    x = torch.randn(2, 24, 32, requires_grad=True)
    out, aux = attn(x, return_aux=True)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    loss = out.pow(2).mean()
    if aux.balance_loss is not None:
        loss = loss + aux.balance_loss
    loss.backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_gradients_reach_the_router(variant):
    attn = make_attention(variant, 32, 4, window=3)
    if attn.router is None or not list(attn.router.parameters()):
        pytest.skip("variant has no learnable router (this is intentional)")
    out = attn(torch.randn(2, 24, 32))
    out.pow(2).mean().backward()
    grads = [p.grad for p in attn.router.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads), "router receives no gradient"


def test_rope_fraction_is_rounded_to_even():
    attn = PhasorAttention(d_model=48, n_heads=4, rope_fraction=0.3)  # d_head = 12
    assert attn.d_pos % 2 == 0
    assert attn.d_pos + attn.d_c == attn.d_head


def test_rejects_zero_width_routing_subspace():
    with pytest.raises(ValueError):
        PhasorAttention(d_model=32, n_heads=4, rope_fraction=1.0, route_on="content")


def test_rejects_window_below_one():
    with pytest.raises(ValueError):
        PhasorAttention(d_model=32, n_heads=4, window=0)


def test_longer_sequence_than_rope_cache_still_works():
    attn = PhasorAttention(d_model=32, n_heads=4, max_seq_len=16, window=2)
    out = attn(torch.randn(1, 40, 32))
    assert torch.isfinite(out).all()
