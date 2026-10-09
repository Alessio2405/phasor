"""Router semantics, and the numerical counterpart of Proposition 1."""

import math

import pytest
import torch

from phasor.router import (
    PhasorRouter,
    bucket_marginal,
    load_balance_entropy_loss,
    normalised_load_entropy,
)


def test_bucket_count_is_twice_content_dim():
    r = PhasorRouter(d_c=16)
    assert r.n_buckets == 32


def test_prototype_min_angle_is_pi_over_two_at_init():
    """Proposition 1, before training."""
    r = PhasorRouter(d_c=12)
    assert r.prototype_min_angle().item() == pytest.approx(math.pi / 2, abs=1e-4)


def test_prototype_min_angle_survives_training():
    """Proposition 1, the version that matters.

    We train the router against an objective that *rewards* collapse -- pushing
    every input into bucket 0 -- and check the prototype geometry is unmoved.
    An unconstrained router would happily deform here; this one cannot.
    """
    r = PhasorRouter(d_c=12)
    opt = torch.optim.Adam(r.parameters(), lr=3e-2)
    x = torch.randn(256, 12)

    for _ in range(150):
        opt.zero_grad()
        s = r.scores(x)  # (256, 1, B)
        loss = -s[..., 0].mean()  # drive everything toward bucket 0
        loss.backward()
        opt.step()

    assert r.prototype_min_angle().item() == pytest.approx(math.pi / 2, abs=1e-3)


def test_isotropic_data_cannot_be_unbalanced_by_the_router():
    """Finding F2 (see NOTES_FOR_PAPER.md), stronger than the paper states.

    An orthogonal map preserves an isotropic distribution, so with isotropic
    routing inputs the load is *exactly* balanced and no amount of gradient
    pressure on ``R`` can concentrate it. This is a sharper guarantee than
    Proposition 1 -- but it is conditional on the data, which is the catch.
    """
    r = PhasorRouter(d_c=8)
    x = torch.randn(4096, 8)  # isotropic
    opt = torch.optim.Adam(r.parameters(), lr=5e-2)

    before = normalised_load_entropy(r(x).bucket, r.n_buckets)
    for _ in range(300):
        opt.zero_grad()
        (-r.scores(x)[..., 0].mean()).backward()  # push everything to bucket 0
        opt.step()
    after = normalised_load_entropy(r(x).bucket, r.n_buckets)

    assert before > 0.95 and after > 0.95, (
        f"isotropic load should stay balanced under collapse pressure, "
        f"got {before:.3f} -> {after:.3f}"
    )


def test_anisotropic_data_collapses_the_load_regardless_of_the_router():
    """The other half of F2, and the paper's honesty check.

    Proposition 1 protects the *codebook*, not the *load*. When the encoder
    concentrates its output directions, the load collapses no matter what ``R``
    does -- so ablation A6's balancing term has to act on the encoder, not on
    the router. This is the single most likely way the method underperforms.
    """
    r = PhasorRouter(d_c=8)
    concentrated = torch.randn(4096, 8) * 0.15 + torch.tensor([3.0] + [0.0] * 7)

    entropy = normalised_load_entropy(r(concentrated).bucket, r.n_buckets)
    n_used = len(r(concentrated).bucket.unique())

    assert entropy < 0.2 and n_used <= 2, (
        f"anisotropic input should collapse the load, got entropy={entropy:.3f}, "
        f"buckets used={n_used}"
    )


def test_bucket_is_in_range_and_deterministic():
    r = PhasorRouter(d_c=8, n_rounds=3)
    x = torch.randn(4, 5, 8)
    out1 = r(x)
    out2 = r(x)
    assert out1.bucket.shape == (4, 5, 3)
    assert out1.bucket.min() >= 0 and out1.bucket.max() < r.n_buckets
    torch.testing.assert_close(out1.bucket, out2.bucket)


def test_routing_is_scale_invariant():
    """Only direction matters: the router normalises before rotating."""
    r = PhasorRouter(d_c=8)
    x = torch.randn(32, 8)
    b1 = r(x).bucket
    b2 = r(x * 7.5).bucket
    torch.testing.assert_close(b1, b2)


def test_scores_are_bounded_by_one():
    r = PhasorRouter(d_c=16)
    s = r.scores(torch.randn(128, 16))
    assert s.abs().max() <= 1.0 + 1e-5


def test_winning_score_lower_bound():
    """Section 4.3: the winning score lives in ``[1/sqrt(d_c), 1]``."""
    d_c = 16
    r = PhasorRouter(d_c=d_c)
    s = r.scores(torch.randn(256, d_c))
    winning = s.max(dim=-1).values
    assert winning.min() >= 1.0 / math.sqrt(d_c) - 1e-4
    assert winning.max() <= 1.0 + 1e-5


def test_gate_is_differentiable_wrt_rotation():
    r = PhasorRouter(d_c=8)
    out = r(torch.randn(16, 8))
    out.gate.sum().backward()
    grads = [p.grad for p in r.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_frozen_router_has_no_gradients():
    r = PhasorRouter(d_c=8, rotation="frozen_random")
    assert list(r.parameters()) == []


def test_balance_loss_extremes():
    """Uniform assignment gives ``-log B``; total collapse gives ``~0``."""
    B = 8
    d_c = B // 2
    uniform = torch.full((1, 1, B), 0.0)
    loss_uniform = load_balance_entropy_loss(uniform, torch.zeros(1, dtype=torch.long), B)
    assert loss_uniform.item() == pytest.approx(-math.log(B), abs=1e-4)

    peaked = torch.full((1, 1, B), -50.0)
    peaked[..., 0] = 50.0
    loss_peaked = load_balance_entropy_loss(peaked, torch.zeros(1, dtype=torch.long), B)
    assert loss_peaked.item() == pytest.approx(0.0, abs=1e-3)
    assert loss_peaked > loss_uniform


def test_bucket_marginal_sums_to_one():
    b = torch.randint(0, 6, (10, 20))
    pi = bucket_marginal(b, 6)
    assert pi.sum().item() == pytest.approx(1.0)
    assert pi.shape == (6,)


def test_normalised_entropy_range():
    B = 16
    balanced = torch.arange(B).repeat(100)
    collapsed = torch.zeros(100 * B, dtype=torch.long)
    assert normalised_load_entropy(balanced, B) == pytest.approx(1.0, abs=1e-6)
    assert normalised_load_entropy(collapsed, B) == pytest.approx(0.0, abs=1e-6)


def test_rejects_zero_content_dim():
    with pytest.raises(ValueError):
        PhasorRouter(d_c=0)
