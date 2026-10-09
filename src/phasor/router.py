"""The PHASOR router (Definition 1 of the paper).

Bucket assignment is ``argmax([R u_hat ; -R u_hat])`` over the *fixed*
cross-polytope codebook ``{+-e_j}``, so ``B = 2 * d_c``.

The fixed codebook is not incidental. With learned prototypes the rotation would
be absorbed (``<R u, c> = <u, R^T c>``) and this would be an ordinary cosine
router. Because the codebook is fixed and ``R`` is constrained to ``O(d_c)``, the
partition geometry is invariant under training: see
:func:`prototype_min_angle`, which is the numerical counterpart of Proposition 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rotation import Rotation, build_rotation


@dataclass
class RouterOutput:
    """Per-round routing results.

    Shapes carry a trailing round axis ``R``:
        ``bucket``: ``(..., R)`` long, in ``[0, B)``
        ``gate``:   ``(..., R)`` float, differentiable w.r.t. the rotation
        ``scores``: ``(..., R, B)`` float, the raw cosine scores in ``[-1, 1]``
    """

    bucket: torch.Tensor
    gate: torch.Tensor
    scores: torch.Tensor


class PhasorRouter(nn.Module):
    """Angular router over a fixed cross-polytope codebook.

    Args:
        d_c: content dimension. The bucket count is ``B = 2 * d_c``.
        rotation: ``"cayley"``, ``"householder"`` or ``"frozen_random"``.
        n_rounds: number of independent rotations, as in Reformer's multi-round
            hashing. The key sets of all rounds are unioned downstream.
        temperature: softmax temperature for the gate.
        straight_through: if True, also expose a straight-through one-hot so that
            gradients reach ``R`` through the assignment itself and not only
            through the gate magnitude. Off by default; ablate it.
    """

    def __init__(
        self,
        d_c: int,
        rotation: str = "cayley",
        n_rounds: int = 1,
        temperature: float = 1.0,
        straight_through: bool = False,
        eps: float = 1e-6,
        rotation_kwargs: dict | None = None,
    ):
        super().__init__()
        if d_c <= 0:
            raise ValueError("d_c must be positive; the router needs a content subspace")
        self.d_c = d_c
        self.n_buckets = 2 * d_c
        self.n_rounds = n_rounds
        self.temperature = temperature
        self.straight_through = straight_through
        self.eps = eps

        kwargs = dict(rotation_kwargs or {})
        self.rotations = nn.ModuleList()
        for r in range(n_rounds):
            rk = dict(kwargs)
            if rotation in ("frozen", "frozen_random", "random") and "seed" not in rk:
                rk["seed"] = 1234 + r
            self.rotations.append(build_rotation(rotation, d_c, **rk))

    # -- geometry -----------------------------------------------------------

    def prototypes(self, round_idx: int = 0) -> torch.Tensor:
        """Effective prototypes ``p_b = R^T c_b``, shape ``(B, d_c)``."""
        R = self.rotations[round_idx].matrix()
        return torch.cat([R, -R], dim=0)

    def prototype_min_angle(self, round_idx: int = 0) -> torch.Tensor:
        """Minimum angle between non-antipodal prototypes, in radians.

        Proposition 1 says this equals ``pi/2`` for every parameter value. This
        is the numerical check; :mod:`tests.test_rotation` asserts it holds after
        optimisation steps, not just at init.
        """
        P = self.prototypes(round_idx).float()
        P = F.normalize(P, dim=-1)
        cos = (P @ P.t()).clamp(-1.0, 1.0)
        B = P.shape[0]
        eye = torch.eye(B, device=P.device, dtype=torch.bool)
        antipodal = cos < -1.0 + 1e-4
        valid = ~(eye | antipodal)
        return torch.acos(cos[valid]).min()

    # -- routing ------------------------------------------------------------

    def scores(self, u: torch.Tensor) -> torch.Tensor:
        """Raw scores ``(..., R, B)`` for content vectors ``u`` of shape ``(..., d_c)``."""
        u_hat = u / (u.norm(dim=-1, keepdim=True) + self.eps)
        per_round = []
        for rot in self.rotations:
            ru = rot.rotate(u_hat)  # (..., d_c)
            per_round.append(torch.cat([ru, -ru], dim=-1))  # (..., B)
        return torch.stack(per_round, dim=-2)  # (..., R, B)

    def forward(self, u: torch.Tensor) -> RouterOutput:
        s = self.scores(u)
        bucket = s.argmax(dim=-1)  # (..., R)

        # The gate is the softmax probability *at the hard index*. This is
        # differentiable w.r.t. the rotation without any estimator: the index is
        # discrete but the probability at that index is not.
        probs = F.softmax(s / self.temperature, dim=-1)
        gate = probs.gather(-1, bucket.unsqueeze(-1)).squeeze(-1)

        if self.straight_through:
            hard = F.one_hot(bucket, self.n_buckets).to(probs.dtype)
            st = hard + probs - probs.detach()
            gate = st.gather(-1, bucket.unsqueeze(-1)).squeeze(-1)

        return RouterOutput(bucket=bucket, gate=gate, scores=s)


# -- balancing ---------------------------------------------------------------


def bucket_marginal(bucket: torch.Tensor, n_buckets: int) -> torch.Tensor:
    """Empirical bucket distribution ``pi``, shape ``(n_buckets,)``.

    ``bucket`` may have any shape; all axes are pooled.
    """
    counts = torch.bincount(bucket.reshape(-1), minlength=n_buckets).float()
    return counts / counts.sum().clamp(min=1.0)


def load_balance_entropy_loss(
    scores: torch.Tensor,
    bucket: torch.Tensor,
    n_buckets: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """``L_bal = sum_b pi_b log pi_b`` (i.e. negative entropy), differentiable.

    The hard marginal has no gradient, so we use the *soft* marginal -- the mean
    softmax assignment -- which is the standard trick and matches the hard
    marginal in expectation.

    Proposition 1 does **not** bound the load, so this term is still required.
    The paper's hypothesis (ablation A6) is only that the required ``lambda`` is
    smaller than for an unconstrained router; that is a hypothesis, not a result.
    """
    probs = F.softmax(scores / temperature, dim=-1)  # (..., R, B)
    pi = probs.reshape(-1, n_buckets).mean(dim=0)
    return (pi * (pi + 1e-9).log()).sum()


def normalised_load_entropy(bucket: torch.Tensor, n_buckets: int) -> float:
    """``H(pi) / log B`` in ``[0, 1]``. 1.0 is perfectly balanced."""
    pi = bucket_marginal(bucket, n_buckets)
    h = -(pi * (pi + 1e-12).log()).sum()
    import math

    return float(h / math.log(n_buckets))
