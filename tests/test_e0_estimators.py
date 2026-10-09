"""The E0 estimators must be right before any E0 number is worth reading.

An estimator bug here would not produce an obviously wrong answer -- it would
produce a plausible one, and the RoPE-decoupling contribution would be accepted
or withdrawn on the strength of it.
"""

import math
import pathlib
import sys

import pytest
import torch

sys.path.insert(
    0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments" / "e0_rope_contamination")
)

from e0_estimators import analyse, entropy, mutual_information, spherical_kmeans  # noqa: E402


def test_mutual_information_of_independent_variables_is_zero():
    g = torch.Generator().manual_seed(0)
    a = torch.randint(0, 8, (20000,), generator=g)
    b = torch.randint(0, 5, (20000,), generator=g)
    assert mutual_information(a, b, 8, 5) == pytest.approx(0.0, abs=0.01)


def test_mutual_information_of_identical_variables_is_entropy():
    g = torch.Generator().manual_seed(0)
    a = torch.randint(0, 8, (20000,), generator=g)
    mi = mutual_information(a, a, 8, 8)
    assert mi == pytest.approx(entropy(a, 8), abs=0.01)
    assert mi == pytest.approx(math.log(8), abs=0.05)


def test_entropy_extremes():
    assert entropy(torch.zeros(100, dtype=torch.long), 4) == pytest.approx(0.0)
    balanced = torch.arange(4).repeat(250)
    assert entropy(balanced, 4) == pytest.approx(math.log(4), abs=1e-6)


def test_spherical_kmeans_recovers_planted_clusters():
    g = torch.Generator().manual_seed(0)
    d, k, n = 32, 6, 1200
    centres = torch.nn.functional.normalize(torch.randn(k, d, generator=g), dim=-1)
    truth = torch.randint(0, k, (n,), generator=g)
    x = centres[truth] + 0.03 * torch.randn(n, d, generator=g)

    assign = spherical_kmeans(x, k, seed=0)
    # Label permutation is arbitrary, so score by mutual information.
    mi = mutual_information(assign, truth, k, k)
    assert mi > 0.95 * entropy(truth, k), "k-means failed to recover planted clusters"


def test_spherical_kmeans_ignores_magnitude():
    """Only direction may matter -- the same invariance the router has."""
    g = torch.Generator().manual_seed(1)
    x = torch.randn(400, 16, generator=g)
    scales = torch.rand(400, 1, generator=g) * 9 + 1
    a = spherical_kmeans(x, 8, seed=3)
    b = spherical_kmeans(x * scales, 8, seed=3)
    assert mutual_information(a, b, 8, 8) > 0.95 * entropy(a, 8)


def test_analyse_separates_content_from_position():
    """The two extremes E0 has to be able to tell apart."""
    g = torch.Generator().manual_seed(0)
    d, T, n_types = 48, 3000, 16

    protos = torch.randn(n_types, d, generator=g)
    tokens = torch.randint(0, n_types, (T,), generator=g)
    content_keys = protos[tokens] + 0.05 * torch.randn(T, d, generator=g)
    positions = torch.arange(T)

    r = analyse(content_keys, positions, tokens, 32)
    assert r["mi_token_norm"] > 0.8
    assert r["mi_position_norm"] < 0.15

    n_bins = 10
    bin_protos = torch.randn(n_bins, d, generator=g)
    bins = (positions.float() / T * n_bins).long().clamp(max=n_bins - 1)
    pos_keys = bin_protos[bins] + 0.05 * torch.randn(T, d, generator=g)

    r = analyse(pos_keys, positions, tokens, 32)
    assert r["mi_position_norm"] > 0.6
    assert r["mi_token_norm"] < 0.15


def test_effective_buckets_is_load_perplexity():
    """Balanced use of ``k`` clusters must report ``k``; total collapse reports 1."""
    g = torch.Generator().manual_seed(0)
    d, T = 16, 2000
    k = 8
    centres = torch.nn.functional.normalize(torch.randn(k, d, generator=g), dim=-1)
    truth = torch.arange(T) % k  # exactly balanced
    x = centres[truth] + 0.02 * torch.randn(T, d, generator=g)

    r = analyse(x, torch.arange(T), truth, k)
    assert r["effective_buckets"] == pytest.approx(k, rel=0.05)
    assert r["load_entropy_norm"] == pytest.approx(1.0, abs=0.05)


def test_rope_degrades_lexical_recoverability():
    """The headline synthetic result, as a regression test.

    Same content vectors, RoPE applied: ``MI(cluster; token)`` must fall. If this
    ever stops holding, the synthetic evidence for Proposition 2 is gone.
    """
    from phasor.rope import apply_rope, build_rope_cache

    g = torch.Generator().manual_seed(0)
    d, T, n_types = 64, 8192, 24
    protos = torch.randn(n_types, d, generator=g)
    tokens = torch.randint(0, n_types, (T,), generator=g)
    keys = protos[tokens] + 0.05 * torch.randn(T, d, generator=g)
    positions = torch.arange(T)

    clean = analyse(keys, positions, tokens, 32)
    cos, sin = build_rope_cache(T, d)
    roped = analyse(apply_rope(keys, cos, sin), positions, tokens, 32)

    assert roped["mi_token_norm"] < clean["mi_token_norm"] - 0.2, (
        f"RoPE did not degrade lexical clustering: "
        f"{clean['mi_token_norm']:.3f} -> {roped['mi_token_norm']:.3f}"
    )
