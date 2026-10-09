"""Estimators shared by `e0_synthetic.py` and `run_e0.py`.

Kept in one place so that the synthetic validation actually validates the code
the real experiment runs. Tested in ``tests/test_e0_estimators.py``.
"""

from __future__ import annotations

import math

import torch


def _kmeanspp_init(x: torch.Tensor, n_clusters: int, generator) -> torch.Tensor:
    """k-means++ seeding on the sphere, with cosine distance ``1 - cos``.

    Plain random seeding regularly draws two centroids from the same true
    cluster and leaves another unrecovered, which shows up as an
    *under*-estimate of the effective bucket count -- precisely the quantity
    Finding F1 asks us to measure. So the seeding is not incidental here.
    """
    n = x.shape[0]
    first = int(torch.randint(n, (1,), generator=generator))
    centroids = [x[first]]
    closest = 1.0 - x @ centroids[0]

    for _ in range(1, n_clusters):
        weights = closest.clamp(min=0) ** 2
        total = weights.sum()
        if total <= 0:
            idx = int(torch.randint(n, (1,), generator=generator))
        else:
            idx = int(torch.multinomial(weights / total, 1, generator=generator))
        centroids.append(x[idx])
        closest = torch.minimum(closest, 1.0 - x @ centroids[-1])

    return torch.stack(centroids)


def spherical_kmeans(
    x: torch.Tensor,
    n_clusters: int,
    n_iter: int = 50,
    seed: int = 0,
    n_restarts: int = 3,
) -> torch.Tensor:
    """Cluster L2-normalised vectors by cosine similarity. Returns assignments.

    Deliberately the same geometry as the routers under study: only direction
    matters, as in Routing Transformer and in :class:`phasor.PhasorRouter`.

    Uses k-means++ seeding and keeps the best of ``n_restarts`` runs by total
    cosine similarity to the assigned centroid.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.nn.functional.normalize(x.float(), dim=-1)
    n = x.shape[0]

    best_assign, best_score = None, -float("inf")
    for _ in range(n_restarts):
        centroids = _kmeanspp_init(x, n_clusters, g)
        assign = torch.full((n,), -1, dtype=torch.long)
        for _ in range(n_iter):
            sims = x @ centroids.t()
            new_assign = sims.argmax(dim=-1)
            if torch.equal(new_assign, assign):
                break
            assign = new_assign
            for c in range(n_clusters):
                sel = assign == c
                if sel.any():
                    centroids[c] = torch.nn.functional.normalize(x[sel].mean(0), dim=-1)
        score = float((x * centroids[assign]).sum())
        if score > best_score:
            best_score, best_assign = score, assign
    return best_assign


def mutual_information(a: torch.Tensor, b: torch.Tensor, na: int, nb: int) -> float:
    """``I(a; b)`` in nats, from hard assignments."""
    joint = torch.zeros(na * nb)
    joint.scatter_add_(0, a * nb + b, torch.ones(a.numel()))
    joint = (joint / joint.sum()).view(na, nb)
    pa = joint.sum(1, keepdim=True)
    pb = joint.sum(0, keepdim=True)
    nz = joint > 0
    ratio = joint[nz] / (pa.expand_as(joint)[nz] * pb.expand_as(joint)[nz])
    return float((joint[nz] * torch.log(ratio)).sum())


def entropy(x: torch.Tensor, n: int) -> float:
    counts = torch.bincount(x, minlength=n).float()
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * p.log()).sum())


def analyse(
    keys: torch.Tensor,
    positions: torch.Tensor,
    token_ids: torch.Tensor,
    n_clusters: int,
    n_pos_bins: int = 16,
    seed: int = 0,
) -> dict:
    """Cluster ``keys`` and measure what the clusters encode.

    Returns normalised mutual informations in ``[0, 1]`` plus the effective
    bucket count -- which Finding F1 argues is the more diagnostic of the two,
    because RoPE contamination shows up as a *shrinking codebook* rather than as
    uniform hashing.
    """
    assign = spherical_kmeans(keys, n_clusters, seed=seed)

    T = int(positions.max()) + 1
    pos_bin = (positions.float() / T * n_pos_bins).long().clamp(max=n_pos_bins - 1)

    uniq_tok, tok_idx = token_ids.unique(return_inverse=True)
    n_tok = len(uniq_tok)

    h_pos = entropy(pos_bin, n_pos_bins)
    h_tok = entropy(tok_idx, n_tok)

    mi_pos = mutual_information(assign, pos_bin, n_clusters, n_pos_bins)
    mi_tok = mutual_information(assign, tok_idx, n_clusters, n_tok)

    counts = torch.bincount(assign, minlength=n_clusters).float()
    p = counts / counts.sum()
    p_nz = p[p > 0]
    load_h = float(-(p_nz * p_nz.log()).sum())

    return {
        "mi_position_norm": mi_pos / h_pos if h_pos > 0 else 0.0,
        "mi_token_norm": mi_tok / h_tok if h_tok > 0 else 0.0,
        "effective_buckets": float(math.exp(load_h)),
        "load_entropy_norm": load_h / math.log(n_clusters),
        "n_clusters": n_clusters,
    }
