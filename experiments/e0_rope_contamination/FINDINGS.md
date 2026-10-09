# E0 — findings

## Status

- [x] **Synthetic validation** (`e0_synthetic.py`) — run, results below.
- [ ] **E0 proper** (`run_e0.py`) — not run. `extract_keys` needs the per-architecture hook.

Nothing here is evidence about a real language model. The synthetic run isolates
the mechanism with all confounds removed, which makes it a lower bound on the
effect and *not* a substitute for E0.

## Synthetic run

`d = 64`, 24 token types, 32 clusters, RoPE base 10 000. Content vectors are
drawn once per token type and reused, so any positional structure in the clusters
comes from rotary phase and from nothing else.

**Estimator validation.** On data with known structure the measures behave:

| construction | MI(pos) | MI(token) | eff. buckets |
|---|---:|---:|---:|
| content-only (truth: lexical) | 0.021 | 1.000 | 29.3 |
| position-only (truth: positional) | 0.782 | 0.054 | 25.3 |

**The mechanism.** Same content, RoPE applied, context growing:

| context | MI(pos) | MI(token) | eff. buckets (of 32) |
|---:|---:|---:|---:|
| 256 | 0.379 | 0.945 | 29.5 |
| 1 024 | 0.159 | 0.940 | 29.1 |
| 4 096 | 0.128 | 0.810 | 31.0 |
| 16 384 | 0.157 | 0.497 | 31.2 |
| 4 096, **no RoPE** | 0.021 | **1.000** | 29.3 |

## What this says

**The effect is real and grows with context.** Lexical recoverability falls from
1.000 to 0.497 as context goes from nothing to 16k — under RoPE, clustering the
*same* content vectors increasingly fails to recover which token they were. That
is Proposition 2's mechanism, isolated.

**MI(position) is the wrong headline metric.** It is non-monotone: 0.379 at
T = 256, falling to 0.128 at T = 4 096, then back to 0.157. At short context the phase has
barely advanced, so position and cluster correlate strongly; at long context the
phase scrambles and the correlation with *binned* position washes out even though
the damage is worse. **MI(token) falls monotonically and is the diagnostic to
report.** The protocol in the paper (Section 5.1) currently leads with
`I(cluster; position)`; it should lead with the loss of `I(cluster; token)`.

**Effective bucket count does not move here, unlike in the router.** In
`tests/test_rope_proposition.py` the argmax router collapses to ~5 of 32 buckets
(Finding F1), but spherical k-means keeps using all 32. The difference is the
partition: k-means fits centroids to wherever the data went, an argmax router
cannot. So F1 is a property of *fixed-codebook* routing specifically, and E0 run
with k-means will not see it. **E0 should therefore be run twice — once with
k-means and once with a fixed-codebook argmax router — or it will measure the
wrong failure.**

## Next

1. Implement `extract_keys` for Pythia/GPT-NeoX. Assert the pre- and post-RoPE
   tensors actually differ before trusting any number.
2. Report MI(token) as the primary axis, MI(position) as secondary.
3. Run both clustering arms (k-means and fixed-codebook argmax).
4. If pre- and post-RoPE come out comparable on real keys, write that here and
   withdraw the RoPE-decoupling contribution from the paper.
