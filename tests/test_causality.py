"""Causality, by autograd, for every variant.

This is the test suite's load-bearing wall. A causal-masking bug in sparse
attention does not announce itself: the loss simply gets better, and every
downstream number is worthless. So we check the property directly -- there must
be no gradient path from input position ``j`` to output position ``i`` for
``j > i`` -- across all ablation arms, because a bug in one arm would silently
decide an ablation.
"""

import pytest
import torch

from phasor.baselines import VARIANTS, make_attention
from phasor.model import GPT, GPTConfig


def _assert_causal(module, T=24, d_model=32, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(1, T, d_model, requires_grad=True)
    out = module(x)

    for i in (0, 1, T // 3, T // 2, T - 2):
        if x.grad is not None:
            x.grad = None
        out[0, i].sum().backward(retain_graph=True)
        future = x.grad[0, i + 1 :].abs().sum().item()
        assert future == 0.0, (
            f"output position {i} depends on inputs after it "
            f"(total |grad| = {future:.3e})"
        )


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_variant_is_causal(variant):
    attn = make_attention(variant, d_model=32, n_heads=4, window=3)
    _assert_causal(attn)


@pytest.mark.parametrize("stride", [1, 2, 4])
def test_causal_across_summary_strides(stride):
    attn = make_attention(
        "phasor", d_model=32, n_heads=4, window=3, summary_stride=stride
    )
    _assert_causal(attn)


@pytest.mark.parametrize("n_rounds", [1, 2, 3])
def test_causal_with_multiple_rounds(n_rounds):
    attn = make_attention("phasor", d_model=32, n_heads=4, window=3, n_rounds=n_rounds)
    _assert_causal(attn)


def test_causal_with_large_window():
    """A window wider than the sequence must not wrap around."""
    attn = make_attention("phasor", d_model=32, n_heads=4, window=1000)
    _assert_causal(attn, T=16)


def test_full_model_is_causal():
    cfg = GPTConfig(
        vocab_size=97, n_layer=2, n_head=4, d_model=32, max_seq_len=64, attn_variant="phasor"
    )
    model = GPT(cfg)
    T = 20
    idx = torch.randint(0, 97, (1, T))

    logits, _, _ = model(idx)
    base = logits.clone()

    # Change the last token; every earlier logit must be bit-identical.
    idx2 = idx.clone()
    idx2[0, -1] = (idx2[0, -1] + 1) % 97
    logits2, _, _ = model(idx2)

    torch.testing.assert_close(base[0, :-1], logits2[0, :-1], atol=1e-5, rtol=1e-4)


def test_perturbation_of_a_middle_future_token():
    """Not just the last token: changing token t must leave logits < t alone."""
    cfg = GPTConfig(
        vocab_size=97, n_layer=2, n_head=4, d_model=32, max_seq_len=64, attn_variant="phasor"
    )
    model = GPT(cfg)
    T, t = 20, 12
    idx = torch.randint(0, 97, (1, T))
    logits, _, _ = model(idx)

    idx2 = idx.clone()
    idx2[0, t] = (idx2[0, t] + 5) % 97
    logits2, _, _ = model(idx2)

    torch.testing.assert_close(logits[0, :t], logits2[0, :t], atol=1e-5, rtol=1e-4)
    assert not torch.allclose(logits[0, t], logits2[0, t]), "perturbation had no effect at all"
