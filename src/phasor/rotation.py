"""Exact orthogonal parametrisations for the PHASOR router.

Proposition 1 requires orthogonality to hold *exactly at every point of
training*, not approximately via a penalty. All three parametrisations here
satisfy that by construction:

- :class:`CayleyRotation`      -- ``R = (I - A)(I + A)^-1``, ``A`` skew-symmetric.
- :class:`HouseholderRotation` -- product of ``p`` reflections, ``O(p*d)`` to apply.
- :class:`FrozenRandomRotation`-- a random orthogonal matrix, never updated.
  This is the Reformer/LSH ablation (A1): same code path, frozen frame.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class Rotation(nn.Module):
    """Base class. Subclasses return an orthogonal ``(d, d)`` matrix."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def matrix(self) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the rotation to ``(..., d)``."""
        return x @ self.matrix().transpose(-1, -2).to(x.dtype)

    def orthogonality_error(self) -> torch.Tensor:
        """``max |R^T R - I|``. Should stay at numerical noise throughout training."""
        R = self.matrix().float()
        return (R.transpose(-1, -2) @ R - torch.eye(self.dim, device=R.device)).abs().max()


class CayleyRotation(Rotation):
    """``R = (I - A)(I + A)^-1`` with ``A = W - W^T`` skew-symmetric.

    Covers the identity component of ``O(d)`` (i.e. ``SO(d)`` minus the
    eigenvalue ``-1`` locus). Cost is ``O(d^3)`` per *update*, amortised over the
    whole batch, so it is negligible at head-content dimensions (32--64).
    """

    def __init__(self, dim: int, init_scale: float = 0.05):
        super().__init__(dim)
        w = torch.randn(dim, dim) * init_scale / math.sqrt(dim)
        self.weight = nn.Parameter(w)

    def matrix(self) -> torch.Tensor:
        W = self.weight
        A = W - W.transpose(-1, -2)
        eye = torch.eye(self.dim, device=W.device, dtype=W.dtype)
        return torch.linalg.solve((eye + A).transpose(-1, -2), (eye - A).transpose(-1, -2)).transpose(-1, -2)


class HouseholderRotation(Rotation):
    """``R = prod_{t<=p} (I - 2 h_t h_t^T / ||h_t||^2)``.

    Orthogonal for any ``{h_t}``. ``p = d`` spans ``O(d)``; ``p < d`` gives a
    cheaper subfamily. Determinant is ``(-1)^p``, which is harmless here: the
    cross-polytope codebook is symmetric under reflection.
    """

    def __init__(self, dim: int, n_reflections: int | None = None):
        super().__init__(dim)
        p = dim if n_reflections is None else n_reflections
        if not 1 <= p <= dim:
            raise ValueError(f"n_reflections must be in [1, {dim}], got {p}")
        self.n_reflections = p
        self.vectors = nn.Parameter(torch.randn(p, dim) / math.sqrt(dim))

    def matrix(self) -> torch.Tensor:
        v = self.vectors
        R = torch.eye(self.dim, device=v.device, dtype=v.dtype)
        for t in range(self.n_reflections):
            h = v[t]
            h = h / (h.norm() + 1e-8)
            R = R - 2.0 * torch.outer(R @ h, h)
        return R

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply reflections sequentially: ``O(p*d)`` instead of materialising ``R``.

        The base class defines ``rotate(x) = x @ R^T``. Each Householder factor
        is symmetric but their product is not, so ``R^T = H_p ... H_1`` and the
        reflections must be applied in reverse order. Getting this backwards
        gives a perfectly valid orthogonal map that simply is not ``R``, which
        would pass every orthogonality test and silently disagree with
        :meth:`matrix`.
        """
        v = self.vectors.to(x.dtype)
        y = x
        for t in reversed(range(self.n_reflections)):
            h = v[t]
            h = h / (h.norm() + 1e-8)
            y = y - 2.0 * (y @ h).unsqueeze(-1) * h
        return y


class FrozenRandomRotation(Rotation):
    """A random orthogonal matrix, registered as a buffer and never trained.

    This is the ablation A1 baseline: Reformer-style random-projection LSH,
    reached by swapping this in for :class:`CayleyRotation` and changing nothing
    else. If PHASOR does not beat this, the central claim of the paper is false
    (cf. the fixed-hash null of Roller et al., 2021).
    """

    def __init__(self, dim: int, seed: int | None = None):
        super().__init__(dim)
        gen = None
        if seed is not None:
            gen = torch.Generator().manual_seed(seed)
        a = torch.randn(dim, dim, generator=gen)
        q, r = torch.linalg.qr(a)
        q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)  # make QR deterministic
        self.register_buffer("_matrix", q)

    def matrix(self) -> torch.Tensor:
        return self._matrix


def build_rotation(kind: str, dim: int, **kwargs) -> Rotation:
    kind = kind.lower()
    if kind == "cayley":
        return CayleyRotation(dim, **kwargs)
    if kind == "householder":
        return HouseholderRotation(dim, **kwargs)
    if kind in ("frozen", "frozen_random", "random"):
        return FrozenRandomRotation(dim, **kwargs)
    raise ValueError(f"unknown rotation kind: {kind!r}")
