#!/usr/bin/env python3
"""Controlled numerical studies for the paper's Section 5.

These are not language-modelling results. They are measurements of the
mechanism, on constructed inputs where the ground truth is known, run against
the reference implementation. Their value is that they are exact, cheap, and
falsifying: two of them have already overturned claims the paper made.

Each study writes a CSV. The paper's tables are generated from these CSVs by
`make_tables.py`, so no number in the paper is typed by hand.

Studies:
  S1  phase contamination vs context length and head width
  S2  effective codebook collapse under RoPE routing
  S3  load as a function of input anisotropy (Proposition 3)
  S4  codebook geometry under collapse pressure (Proposition 1)
  S5  pattern quality: recall of dense top-k mass vs pattern density
  S6  residue branch calibration vs bias initialisation
  S7  chunked-vs-masked pattern divergence
"""

from __future__ import annotations

import csv
import math
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phasor.attention import PhasorAttention  # noqa: E402
from phasor.baselines import make_attention  # noqa: E402
from phasor.metrics import pattern_density, topk_mass_recall  # noqa: E402
from phasor.rope import apply_rope, build_rope_cache  # noqa: E402
from phasor.router import PhasorRouter, normalised_load_entropy  # noqa: E402

OUT = pathlib.Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def write_csv(name: str, rows: list[dict]) -> None:
    path = OUT / f"{name}.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path.name}  ({len(rows)} rows)")


def agreement(buckets: torch.Tensor) -> float:
    return float((buckets == buckets[0]).float().mean())


# ---------------------------------------------------------------------------


def s1_phase_contamination(n_trials: int = 32) -> list[dict]:
    """Co-routing agreement for content-identical tokens, vs context and width.

    A token's content is held fixed and only its position varies, so any loss of
    co-routing is attributable to rotary phase.
    """
    rows = []
    for d_c in (8, 16, 32, 64):
        router = PhasorRouter(d_c=d_c)
        B = router.n_buckets
        for T in (32, 128, 512, 2048, 8192, 32768):
            cos, sin = build_rope_cache(T, d_c)
            torch.manual_seed(0)
            vals = []
            for _ in range(n_trials):
                u = torch.randn(d_c)
                rep = u.unsqueeze(0).expand(T, d_c)
                vals.append(agreement(router(apply_rope(rep, cos, sin)).bucket.squeeze(-1)))
            rows.append(
                dict(
                    d_c=d_c,
                    n_buckets=B,
                    context=T,
                    agreement=round(sum(vals) / len(vals), 4),
                    chance=round(1.0 / B, 4),
                    ratio_to_chance=round((sum(vals) / len(vals)) * B, 2),
                )
            )
    return rows


def s2_effective_codebook(n_trials: int = 32) -> list[dict]:
    """How much of the codebook an angular router actually reaches under RoPE.

    Reported as the perplexity of the visited-bucket distribution, which is the
    quantity Finding F1 argues the protocol should track.
    """
    rows = []
    for d_c in (8, 16, 32, 64):
        router = PhasorRouter(d_c=d_c)
        B = router.n_buckets
        for T in (512, 2048, 8192, 32768):
            cos, sin = build_rope_cache(T, d_c)
            torch.manual_seed(0)
            uniq, perp = [], []
            for _ in range(n_trials):
                u = torch.randn(d_c)
                b = router(apply_rope(u.unsqueeze(0).expand(T, d_c), cos, sin)).bucket.squeeze(-1)
                uniq.append(len(b.unique()))
                counts = torch.bincount(b, minlength=B).float()
                p = counts / counts.sum()
                p = p[p > 0]
                perp.append(float(torch.exp(-(p * p.log()).sum())))
            rows.append(
                dict(
                    d_c=d_c,
                    n_buckets=B,
                    context=T,
                    buckets_visited=round(sum(uniq) / len(uniq), 2),
                    effective_buckets=round(sum(perp) / len(perp), 2),
                    fraction_of_codebook=round(sum(perp) / len(perp) / B, 3),
                )
            )
    return rows


def s3_load_vs_anisotropy(n: int = 8192, steps: int = 300) -> list[dict]:
    """Proposition 3, and its boundary.

    Inputs interpolate between isotropic and concentrated. For each, we measure
    load entropy at initialisation and after gradient pressure explicitly
    pushing every token into one bucket. If Proposition 3 holds, the isotropic
    end is immovable.
    """
    rows = []
    d_c = 16
    for conc in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        torch.manual_seed(0)
        direction = torch.zeros(d_c)
        direction[0] = conc
        x = torch.randn(n, d_c) + direction

        router = PhasorRouter(d_c=d_c)
        B = router.n_buckets
        h_before = normalised_load_entropy(router(x).bucket, B)

        opt = torch.optim.Adam(router.parameters(), lr=5e-2)
        for _ in range(steps):
            opt.zero_grad()
            (-router.scores(x)[..., 0].mean()).backward()
            opt.step()
        h_after = normalised_load_entropy(router(x).bucket, B)

        rows.append(
            dict(
                concentration=conc,
                n_buckets=B,
                load_entropy_init=round(h_before, 4),
                load_entropy_after_pressure=round(h_after, 4),
                delta=round(h_after - h_before, 4),
                buckets_used=len(router(x).bucket.unique()),
            )
        )
    return rows


def s4_codebook_geometry(steps: int = 400) -> list[dict]:
    """Proposition 1, tested against what it actually asserts.

    The claim is a statement about reachability: for an unconstrained router the
    minimum prototype angle *can* be driven to zero by gradient descent, and for
    an orthogonal router with a fixed codebook it cannot, for any parameter
    value. So the right test is adversarial -- optimise directly against the
    quantity -- rather than hoping a task-shaped loss happens to collapse it.

    We maximise mean pairwise prototype cosine similarity in both families and
    watch the minimum angle.
    """
    rows = []
    d_c = 16

    def record(name, step, angle):
        rows.append(
            dict(router=name, step=step, min_prototype_angle_deg=round(angle, 3))
        )

    # --- orthogonal router: the adversary gets the rotation parameters --------
    torch.manual_seed(0)
    router = PhasorRouter(d_c=d_c)
    B = router.n_buckets
    opt = torch.optim.Adam(router.parameters(), lr=5e-2)
    eye_B = torch.eye(B, dtype=torch.bool)

    def ortho_angle():
        with torch.no_grad():
            return math.degrees(float(router.prototype_min_angle()))

    record("orthogonal (ours)", 0, ortho_angle())
    for s in range(1, steps + 1):
        opt.zero_grad()
        P = torch.nn.functional.normalize(router.prototypes(), dim=-1)
        # Exclude the antipodal pairs, which are -1 by construction.
        cos = (P @ P.t()).clamp(-1, 1)
        off = cos[~eye_B]
        (-off[off > -0.99].mean()).backward()
        opt.step()
        if s in (50, 150, steps):
            record("orthogonal (ours)", s, ortho_angle())

    # --- unconstrained linear router: same adversary, free prototypes ---------
    torch.manual_seed(0)
    W = torch.nn.Parameter(torch.randn(d_c, B) / math.sqrt(d_c))
    opt = torch.optim.Adam([W], lr=5e-2)

    def free_angle():
        with torch.no_grad():
            P = torch.nn.functional.normalize(W.t(), dim=-1)
            cos = (P @ P.t()).clamp(-1, 1)
            return math.degrees(float(torch.acos(cos[~eye_B]).max().new_tensor(
                float(torch.acos(cos[~eye_B]).min()))))

    record("unconstrained linear", 0, free_angle())
    for s in range(1, steps + 1):
        opt.zero_grad()
        P = torch.nn.functional.normalize(W.t(), dim=-1)
        cos = (P @ P.t()).clamp(-1, 1)
        (-cos[~eye_B].mean()).backward()
        opt.step()
        if s in (50, 150, steps):
            record("unconstrained linear", s, free_angle())

    return rows


def _structured_input(T: int, d_model: int, n_types: int, seed: int) -> torch.Tensor:
    """A sequence drawn from ``n_types`` recurring token embeddings.

    Gaussian noise is the wrong input for this study: it has no content
    structure for a content router to find, so every routing rule scores the
    same and the measurement is vacuous (we checked -- see the paper's
    discussion of the null). Real text repeats token types across distant
    positions, which is the structure the router is supposed to exploit.
    """
    g = torch.Generator().manual_seed(seed)
    vocab = torch.randn(n_types, d_model, generator=g)
    ids = torch.randint(0, n_types, (T,), generator=g)
    return (vocab[ids] + 0.1 * torch.randn(T, d_model, generator=g)).unsqueeze(0)


def s5_pattern_quality(n_trials: int = 8) -> list[dict]:
    """Recall of dense top-k attention mass, against the density that buys it.

    The router is untrained, so this measures the *geometry* the architecture
    imposes, not what training could make of it. The reference point is a random
    pattern of equal density, whose expected recall equals its density; a ratio
    above 1 means the routing rule is finding something.

    Both input regimes are reported, because the contrast is the finding: with
    unstructured input every variant is identical.
    """
    rows = []
    d_model, n_heads = 128, 4
    variants = ("phasor", "a1_frozen_rotation", "a3_route_on_rope", "a4_pure_skip")
    for regime in ("gaussian", "structured"):
        for variant in variants:
            for T in (256, 1024):
                for window in (1, 8):
                    recs, dens = [], []
                    for trial in range(n_trials):
                        torch.manual_seed(trial)
                        attn = make_attention(variant, d_model, n_heads, window=window)
                        if regime == "gaussian":
                            x = torch.randn(1, T, d_model)
                        else:
                            x = _structured_input(T, d_model, n_types=32, seed=trial)
                        recs.append(topk_mass_recall(attn, x, k=32))
                        dens.append(pattern_density(attn, x))
                    r = sum(recs) / len(recs)
                    d = sum(dens) / len(dens)
                    rows.append(
                        dict(
                            regime=regime,
                            variant=variant,
                            context=T,
                            window=window,
                            density=round(d, 4),
                            topk_recall=round(r, 4),
                            recall_over_density=round(r / d if d > 0 else 0.0, 3),
                        )
                    )
    return rows


def s6_residue_calibration() -> list[dict]:
    """How much attention mass the residue branch takes, vs its bias at init.

    Finding F3: at zero bias the branch swallows about half the mass, which puts
    prediction P3's 2--25% band outside the reachable range before training has
    started. The bias is a free hyperparameter and the paper must say what it is.
    """
    rows = []
    d_model, n_heads, T = 128, 4, 512
    for bias in (0.0, -2.0, -4.0, -6.0, -8.0):
        masses = []
        for trial in range(4):
            torch.manual_seed(trial)
            attn = PhasorAttention(
                d_model, n_heads, window=8, summary_bias_init=bias
            )
            _, aux = attn(torch.randn(1, T, d_model), return_aux=True)
            masses.append(aux.summary_mass)
        rows.append(
            dict(
                summary_bias_init=bias,
                summary_mass=round(sum(masses) / len(masses), 4),
                within_P3_band=bool(0.02 <= sum(masses) / len(masses) <= 0.25),
            )
        )
    return rows


def s7_chunk_divergence() -> list[dict]:
    """How far a fixed-size-chunk approximation is from true bucket-wise attention.

    The paper's O(T^1.5) depends on sorting tokens by bucket and attending within
    fixed-size chunks. That is not the same function as attending within buckets:
    a bucket larger than the chunk loses intra-bucket pairs, and a smaller one is
    padded with a neighbour's tokens. We measure the gap directly on the pattern,
    which does not require implementing the fast kernel.

    Reported as: the fraction of true in-bucket causal pairs the chunked pattern
    keeps (recall), and the fraction of the chunked pattern's pairs that are not
    in-bucket at all (contamination).
    """
    rows = []
    d_model, n_heads = 128, 4
    for T in (256, 1024, 4096):
        for chunk in (32, 64, 128):
            recalls, contams, loads = [], [], []
            for trial in range(4):
                torch.manual_seed(trial)
                attn = make_attention("phasor", d_model, n_heads, window=1)
                x = torch.randn(1, T, d_model)

                from phasor.rope import apply_partial_rope, build_rope_cache as brc

                cos, sin = brc(T, attn.d_pos, dtype=x.dtype)
                q = attn._heads(attn.q_proj(x))
                k = attn._heads(attn.k_proj(x))
                qf, qc = apply_partial_rope(q, attn.d_pos, cos, sin)
                kf, kc = apply_partial_rope(k, attn.d_pos, cos, sin)
                bq = attn.router(attn._routing_input(qf, qc)).bucket[0, :, :, 0]
                bk = attn.router(attn._routing_input(kf, kc)).bucket[0, :, :, 0]

                B = attn.router.n_buckets
                counts = torch.bincount(bk[0], minlength=B).float()
                p = counts / counts.sum()
                p = p[p > 0]
                loads.append(float(-(p * p.log()).sum() / math.log(B)))

                pos = torch.arange(T)
                causal = pos.view(-1, 1) >= pos.view(1, -1)

                for h in range(n_heads):
                    true_pat = (bq[h].view(-1, 1) == bk[h].view(1, -1)) & causal

                    # Sort keys by (bucket, position), cut into fixed chunks, and
                    # let a query see the chunk its own bucket-sorted position
                    # falls in, plus the previous one (Reformer's rule).
                    order = torch.argsort(bk[h] * T + pos)
                    rank = torch.empty_like(order)
                    rank[order] = torch.arange(T)
                    q_order = torch.argsort(bq[h] * T + pos)
                    q_rank = torch.empty_like(q_order)
                    q_rank[q_order] = torch.arange(T)

                    q_chunk = q_rank // chunk
                    k_chunk = rank // chunk
                    chunked = (
                        (q_chunk.view(-1, 1) == k_chunk.view(1, -1))
                        | (q_chunk.view(-1, 1) - 1 == k_chunk.view(1, -1))
                    ) & causal

                    tp = (true_pat & chunked).sum().item()
                    recalls.append(tp / max(true_pat.sum().item(), 1))
                    contams.append(
                        1.0 - tp / max(chunked.sum().item(), 1)
                    )
            rows.append(
                dict(
                    context=T,
                    chunk_size=chunk,
                    load_entropy=round(sum(loads) / len(loads), 3),
                    inbucket_recall=round(sum(recalls) / len(recalls), 4),
                    contamination=round(sum(contams) / len(contams), 4),
                )
            )
    return rows


STUDIES = [
    ("s1_phase_contamination", s1_phase_contamination),
    ("s2_effective_codebook", s2_effective_codebook),
    ("s3_load_vs_anisotropy", s3_load_vs_anisotropy),
    ("s4_codebook_geometry", s4_codebook_geometry),
    ("s5_pattern_quality", s5_pattern_quality),
    ("s6_residue_calibration", s6_residue_calibration),
    ("s7_chunk_divergence", s7_chunk_divergence),
]


def main():
    only = sys.argv[1:] or None
    for name, fn in STUDIES:
        if only and not any(o in name for o in only):
            continue
        print(f"[{name}]")
        write_csv(name, fn())
    print("\ndone")


if __name__ == "__main__":
    main()
