"""The toy model trains, and the ablation arms are genuinely matched."""

import pytest
import torch

from phasor.baselines import VARIANTS
from phasor.model import GPT, GPTConfig


def _cfg(variant, **kw):
    return GPTConfig(
        vocab_size=101,
        n_layer=2,
        n_head=4,
        d_model=32,
        max_seq_len=64,
        attn_variant=variant,
        attn_kwargs=dict(window=3, **kw),
    )


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_forward_backward(variant):
    model = GPT(_cfg(variant))
    idx = torch.randint(0, 101, (2, 24))
    logits, loss, stats = model(idx, targets=idx)
    assert logits.shape == (2, 24, 101)
    assert torch.isfinite(loss)
    (loss + stats["aux_loss"]).backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_loss_decreases_on_a_memorisable_batch():
    """Sanity: the thing can learn. Not a result, just a smoke test."""
    torch.manual_seed(0)
    model = GPT(_cfg("phasor"))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    idx = torch.randint(0, 101, (2, 24))

    _, first, _ = model(idx, targets=idx)
    for _ in range(60):
        opt.zero_grad()
        _, loss, stats = model(idx, targets=idx)
        (loss + stats["aux_loss"]).backward()
        opt.step()
    _, last, _ = model(idx, targets=idx)

    assert last < first * 0.7, f"loss went {first.item():.3f} -> {last.item():.3f}"


def test_ablation_arms_have_matched_parameter_counts():
    """A1, A3 and A4 must not confound the thing they test with model size.

    Arms that differ only in routing target, fallback or rotation freezing
    should differ by at most the rotation's own parameters, which we report
    rather than hide.
    """
    counts = {}
    for variant in ("phasor", "a3_route_on_rope", "a4_pure_skip", "a1_frozen_rotation"):
        counts[variant] = GPT(_cfg(variant)).n_params()

    base = counts["phasor"]
    # a3 routes on a different subspace of the same width (rope_fraction=0.5).
    assert counts["a3_route_on_rope"] == base
    # a4 drops the summary scale/bias: 2 scalars per head per layer.
    assert base - counts["a4_pure_skip"] == 2 * 4 * 2  # 2 scalars x n_heads x n_layer
    # a1 freezes the rotation, so it loses exactly the rotation's parameters.
    assert counts["a1_frozen_rotation"] < base


def test_dense_and_phasor_have_comparable_size():
    dense = GPT(_cfg("dense")).n_params()
    phasor = GPT(_cfg("phasor")).n_params()
    overhead = (phasor - dense) / dense
    assert overhead < 0.02, (
        f"router overhead is {overhead:.2%} of non-embedding params; "
        "FLOP-matching in the protocol assumes this is negligible"
    )


def test_stats_are_reported():
    model = GPT(_cfg("phasor"))
    _, _, stats = model(torch.randint(0, 101, (1, 32)))
    for key in ("load_entropy", "summary_mass", "pattern_density"):
        assert stats[key] is not None, f"missing diagnostic {key}"


def test_dense_variant_reports_no_routing_stats():
    model = GPT(_cfg("dense"))
    _, _, stats = model(torch.randint(0, 101, (1, 32)))
    assert stats["load_entropy"] is None
    assert float(stats["aux_loss"]) == 0.0
