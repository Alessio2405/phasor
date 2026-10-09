"""Orthogonality must hold exactly, and must keep holding under optimisation.

Proposition 1 is only worth anything if ``R`` is orthogonal at every point of
training, not merely at initialisation. A penalty-based "soft orthogonality"
would fail these tests, which is why the parametrisations are exact.
"""

import pytest
import torch

from phasor.rotation import (
    CayleyRotation,
    FrozenRandomRotation,
    HouseholderRotation,
    build_rotation,
)

KINDS = ["cayley", "householder", "frozen_random"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dim", [4, 16, 32])
def test_orthogonal_at_init(kind, dim):
    rot = build_rotation(kind, dim)
    assert rot.orthogonality_error() < 1e-4


@pytest.mark.parametrize("kind", ["cayley", "householder"])
def test_orthogonal_after_optimisation(kind):
    """The test that actually matters: 200 Adam steps with a loss that *wants*
    to break orthogonality, and the constraint still holds."""
    dim = 16
    rot = build_rotation(kind, dim)
    opt = torch.optim.Adam(rot.parameters(), lr=1e-2)
    target = torch.randn(dim, dim)  # not orthogonal; pulls R off the manifold

    for _ in range(200):
        opt.zero_grad()
        loss = (rot.matrix() - target).pow(2).mean()
        loss.backward()
        opt.step()

    assert rot.orthogonality_error() < 1e-4, "rotation drifted off O(d) during training"


@pytest.mark.parametrize("kind", KINDS)
def test_rotate_matches_matrix(kind):
    """``rotate`` (the O(p*d) path) must agree with materialising ``R``."""
    dim, n = 16, 7
    rot = build_rotation(kind, dim)
    x = torch.randn(n, dim)
    via_matrix = x @ rot.matrix().t()
    torch.testing.assert_close(rot.rotate(x), via_matrix, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("kind", KINDS)
def test_norm_preserved(kind):
    dim = 24
    rot = build_rotation(kind, dim)
    x = torch.randn(64, dim)
    torch.testing.assert_close(rot.rotate(x).norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-4)


def test_frozen_rotation_has_no_parameters():
    """A1's negative arm must genuinely not learn, or the ablation is vacuous."""
    rot = FrozenRandomRotation(16, seed=0)
    assert list(rot.parameters()) == []


def test_frozen_rotation_is_reproducible():
    a = FrozenRandomRotation(16, seed=7).matrix()
    b = FrozenRandomRotation(16, seed=7).matrix()
    torch.testing.assert_close(a, b)


def test_householder_reflection_count_is_respected():
    rot = HouseholderRotation(16, n_reflections=4)
    assert rot.n_reflections == 4
    assert rot.orthogonality_error() < 1e-4


def test_cayley_gradient_flows():
    rot = CayleyRotation(8)
    x = torch.randn(4, 8)
    rot.rotate(x).sum().backward()
    assert rot.weight.grad is not None
    assert rot.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("dim", [4, 8])
def test_householder_bad_reflection_count_rejected(dim):
    with pytest.raises(ValueError):
        HouseholderRotation(dim, n_reflections=dim + 1)
    with pytest.raises(ValueError):
        HouseholderRotation(dim, n_reflections=0)
