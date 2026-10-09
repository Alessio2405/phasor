#!/usr/bin/env python3
"""E0 on synthetic data: validates the estimators, and isolates the mechanism.

Run this before `run_e0.py`. It does two things:

1. **Validates the estimator.** On data with known structure -- clusters that
   are purely content-driven, and clusters that are purely position-driven -- the
   mutual-information and effective-bucket measures must give the right answer.
   If they do not, the numbers from the real model mean nothing.

2. **Isolates the mechanism.** Real key vectors confound content structure with
   positional structure. Here we control both, so what we see is attributable to
   rotary phase and nothing else.

No language model, no network access, no GPU. Runs in seconds.
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from e0_estimators import analyse  # noqa: E402
from phasor.rope import apply_rope, build_rope_cache  # noqa: E402


def make_content_only(T: int, d: int, n_types: int, seed: int = 0):
    """``n_types`` token types, each a fixed direction, placed at random positions.

    Ground truth: clusters should recover token type and know nothing about
    position.
    """
    g = torch.Generator().manual_seed(seed)
    prototypes = torch.randn(n_types, d, generator=g)
    tokens = torch.randint(0, n_types, (T,), generator=g)
    keys = prototypes[tokens] + 0.05 * torch.randn(T, d, generator=g)
    return keys, tokens, torch.arange(T)


def make_position_only(T: int, d: int, n_bins: int, seed: int = 0):
    """Direction determined by position bin, content random. The opposite case."""
    g = torch.Generator().manual_seed(seed)
    prototypes = torch.randn(n_bins, d, generator=g)
    positions = torch.arange(T)
    bins = (positions.float() / T * n_bins).long().clamp(max=n_bins - 1)
    keys = prototypes[bins] + 0.05 * torch.randn(T, d, generator=g)
    tokens = torch.randint(0, 50, (T,), generator=g)
    return keys, tokens, positions


def make_content_then_rope(T: int, d: int, n_types: int, base: float = 10000.0, seed: int = 0):
    """Content-only keys, then RoPE applied. This is the experiment.

    Any positional information the clusters acquire is attributable to rotary
    phase, because there was none in the content.
    """
    keys, tokens, positions = make_content_only(T, d, n_types, seed)
    cos, sin = build_rope_cache(T, d, base=base)
    return apply_rope(keys, cos, sin), tokens, positions


def main():
    d, n_types, B = 64, 24, 32
    rows = []

    print("=" * 74)
    print("1. Estimator validation on data with known structure")
    print("=" * 74)
    print(f"{'case':<28} {'MI(pos)':>9} {'MI(token)':>10} {'eff.buckets':>12}")
    print("-" * 74)

    keys, tokens, pos = make_content_only(4096, d, n_types)
    r = analyse(keys, pos, tokens, B)
    print(f"{'content-only (truth: lexical)':<28} {r['mi_position_norm']:>9.3f} "
          f"{r['mi_token_norm']:>10.3f} {r['effective_buckets']:>12.1f}")
    assert r["mi_token_norm"] > r["mi_position_norm"], "estimator failed the content case"
    rows.append(dict(case="content_only", **r))

    keys, tokens, pos = make_position_only(4096, d, 12)
    r = analyse(keys, pos, tokens, B)
    print(f"{'position-only (truth: pos.)':<28} {r['mi_position_norm']:>9.3f} "
          f"{r['mi_token_norm']:>10.3f} {r['effective_buckets']:>12.1f}")
    assert r["mi_position_norm"] > r["mi_token_norm"], "estimator failed the position case"
    rows.append(dict(case="position_only", **r))

    print("\nEstimator behaves correctly on both extremes.\n")

    print("=" * 74)
    print("2. The mechanism: content-only keys, RoPE applied, context growing")
    print("=" * 74)
    print(f"{'context':>8} {'MI(pos)':>9} {'MI(token)':>10} {'eff.buckets':>12}  (of 32)")
    print("-" * 74)

    for T in (256, 1024, 4096, 16384):
        keys, tokens, pos = make_content_then_rope(T, d, n_types)
        r = analyse(keys, pos, tokens, B)
        print(f"{T:>8} {r['mi_position_norm']:>9.3f} {r['mi_token_norm']:>10.3f} "
              f"{r['effective_buckets']:>12.1f}")
        rows.append(dict(case="content_then_rope", context=T, **r))

    print()
    print("Reference: the same content WITHOUT RoPE, at T = 4096")
    keys, tokens, pos = make_content_only(4096, d, n_types)
    r = analyse(keys, pos, tokens, B)
    print(f"{4096:>8} {r['mi_position_norm']:>9.3f} {r['mi_token_norm']:>10.3f} "
          f"{r['effective_buckets']:>12.1f}")
    rows.append(dict(case="content_no_rope", context=4096, **r))

    out = pathlib.Path(__file__).parent / "e0_synthetic_results.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")
    print(
        "\nRead this as a lower bound on the effect, not as E0 itself: real keys\n"
        "carry positional structure of their own, which this construction removes."
    )


if __name__ == "__main__":
    main()
