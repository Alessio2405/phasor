# PHASOR

**Phase-Aware Sparse Attention with Learned Orthogonal Routing**

Reference implementation, test suite and the controlled numerical studies behind
the whitepaper in [`paper/`](paper/) (30 pages).

> **No language model has been trained.** Every language-modelling table in the
> paper is a declared placeholder. The only real numbers in this repository come
> from the test suite and from the seven numerical studies, and they are labelled
> as such.

```bash
pip install -e ".[dev]"

pytest tests/ -q                                   # 128 tests
python experiments/numerical_studies.py            # the seven studies
python experiments/make_tables.py                  # regenerate tables + figures
python experiments/e0_rope_contamination/e0_synthetic.py
./experiments/run_ablation_sweep.sh                # toy-scale ablations

cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Every number in Section 5 of the paper comes from these CSVs via
`make_tables.py`. None is typed by hand, so the evidence regenerates end to end.

## The three design choices

| | |
|---|---|
| **Learned orthogonal router** | A learned rotation `R ∈ O(d_c)` acting on a **fixed** cross-polytope codebook. Proposition 1 shows the bucket prototypes stay exactly π/2 apart for every parameter value; `test_prototype_min_angle_survives_training` checks it after optimisation. The fixed codebook is essential — with free prototypes the rotation is absorbed (`⟨Rû, c⟩ = ⟨û, Rᵀc⟩`) and this is just a cosine router. |
| **Decoupling routing from RoPE** | The router reads only the content dimensions. Proposition 2 shows angular routing on rotary dimensions loses content-invariance as phase accumulates; `test_rope_proposition.py` verifies it numerically. |
| **Attention-visible fallback** | Tokens a query's bucket excludes are pooled into `B` causal summaries inside the *same* softmax, instead of Mixture-of-Depths' pure skip. Under a pure skip the excluded token is absent from the softmax, so "attention would filter it anyway" is false; this makes it true. |

## What is not new

Table 4 of the paper states the provenance of every component. Angular routing is
Reformer and Routing Transformer; cosine routing against collapse is X-MoE; the
residual carry of unrouted tokens is Mixture-of-Depths. The claims are limited to
the constrained orthogonal frame, the RoPE argument, and the visibility of the
residue.

## Two overclaims to avoid

Section 4.2 of the paper dismantles both, because both are common in this area:

- Routing **which projection matrix** produces a key does **not** reduce the
  quadratic term. Only the sparsity *pattern* does.
- Sparse attention does **not** reduce KV-cache **capacity** — every token still
  emits one key. It reduces read bandwidth per decoding step. Capacity needs
  MQA/GQA, latent compression or eviction, which are orthogonal and composable.

## What the code already found

Seven findings, all reproducible, all documented in
[`NOTES_FOR_PAPER.md`](NOTES_FOR_PAPER.md). Three of them corrected claims the
paper was making.

- **F1 — the aggregate limit is not `1/B`.** With RoPE's geometric spectrum only
  the high-frequency planes equidistribute, so the argmax stays confined to
  **5–10 effective buckets regardless of `B`** — 37% of a 16-bucket codebook but
  only 8% of a 128-bucket one. Provisioning more capacity makes the relative
  collapse *worse*. Proposition 4 explains it: over positions the routing input
  is confined to a torus of dimension `d_c/2`, a measure-zero subset of the
  sphere.
- **F2 — the load belongs to the data, not the router.** With isotropic input the
  load is exactly balanced and 300 Adam steps of explicit collapse pressure move
  it from 0.999 to 0.998. With anisotropic input it collapses to 1 bucket in 16
  *at initialisation*. This is now Proposition 3 — and it means **ablation A6 as
  originally specified could not work**, since a balancing loss reaching only `R`
  cannot change a marginal that does not depend on `R`.
- **F3 — the residue branch takes 62% of the attention mass** at zero bias and
  0.4% at bias −6: two orders of magnitude on a hyperparameter the paper never
  declared, which made prediction P3 unfalsifiable. Now defaults to −4.
- **F4 — a transpose bug** in `HouseholderRotation.rotate` that passed every
  orthogonality test, because it produced a valid orthogonal map that simply was
  not `R`.
- **F5 — chunking is faithful only when chunk ≈ mean bucket size.** At n=4096
  with chunk 32 the chunked pattern keeps **38%** of true in-bucket pairs, with
  load entropy 0.996 — so this is intrinsic, not an imbalance artefact. It
  follows that `O(n^1.5)` holds only if `d_c ≈ √n/2`: the routing width is
  coupled to the context length. This is the design's most serious limitation.
- **F6 — at initialisation every ablation arm is identical** (max recall spread
  0.011). Cross-polytope bucketing beats a random pattern of equal density by
  2.5–4.7×, but *which* angular rule you use is invisible before training. A1 and
  A3 cannot be settled without training runs.
- **F7 — the training budget decides the answer.** The toy task is learned by a
  phase transition whose onset varies by more than 2× across seeds of an
  identical configuration (step 400 vs 900). A budget truncated near that onset
  makes every slower arm report at the floor, and the comparison *looks* decisive
  when it is only cut short. Report trajectories, not final numbers.

## What is missing, and why it matters

`PhasorAttention` builds the full `(T,T)` score matrix and applies routing as a
mask: it is `O(T²)`. **Good for semantics, says nothing about speed.** The
Reformer-style chunked path is deliberately not implemented — but F5 now
quantifies the gap on the *pattern*, which is measurable without the fast kernel,
and the gap is large. Any wall-clock number must be reported alongside in-bucket
recall, or it measures the speed of a different function from the one the method
defines.

## Layout

```
paper/                          main.tex + sections, refs.bib, PHASOR_whitepaper.pdf
  generated/                    tables and figures, produced by make_tables.py
src/phasor/
  rotation.py                   Cayley, Householder, frozen random rotation
  router.py                     Definition 1, gating, balancing loss
  rope.py                       partial RoPE, content/position split
  summaries.py                  causal per-bucket summaries + decoding state
  attention.py                  the layer (Algorithm 1)
  baselines.py                  every ablation arm from the same class
  metrics.py                    top-k recall, pattern density, load entropy
  model.py                      minimal GPT for the toy experiments
tests/                          128 tests
experiments/
  numerical_studies.py          the seven studies behind Section 5
  make_tables.py                CSV -> LaTeX tables and figures
  e0_rope_contamination/        E0: synthetic (run) + real model (to do)
  train_toy.py                  toy-scale training harness
  run_ablation_sweep.sh         A1/A3/A4 sweep
```

## Conventions for contributors

1. **No invented numbers.** Placeholders in the paper are marked `[--]`. Any
   number added must come from a run with its seed and hardware recorded.
2. **The causality tests are not negotiable.** A causal-masking bug in sparse
   attention does not announce itself — the loss simply improves. If a test in
   `tests/test_causality.py` fails, the code is wrong, never the test.
3. **If a finding contradicts the paper, update the paper.** It has happened
   seven times. Write it into `NOTES_FOR_PAPER.md` rather than working around it
   in code.
4. **One ablation arm = one flag.** Every baseline comes from `PhasorAttention`
   via `baselines.VARIANTS`. A separately written baseline would differ in a
   dozen incidental ways and the ablation would measure those instead.
5. **Section 5 tables are generated, not written.** If you add a study, add its
   generator to `make_tables.py` too.
6. **Report trajectories, not final numbers** (F7), and run every arm until the
   dense baseline has *plateaued* — not merely until it has started to learn.

## Where to go next

1. **E0 on a real model.** `run_e0.py` needs the hook that extracts keys both
   before and after RoPE for the chosen architecture. Report `MI(token)` as the
   primary axis and the effective bucket count, and use **both** clustering
   rules — spherical k-means and fixed-codebook argmax — because the F1 collapse
   is a property of the second and k-means does not see it.
2. **Decouple `B` from `d_c`.** F5 means `O(n^1.5)` requires `d_c ≈ √n/2`.
   Composing several cross-polytope partitions over disjoint subspaces would
   multiply the effective codebook without widening any single rotation. This is
   the most promising design change and it is not in the paper.
3. **A6′:** whitening the routing input, since by Proposition 3 router-side
   balancing cannot work.
4. **The chunked path**, always reporting in-bucket recall next to wall-clock.
5. **Finish the ablation sweep** with a long enough budget (see F7). Only `dense`
   and `phasor` have completed; `run_ablation_sweep.sh` resumes and skips what is
   already done.

## Falsifiable predictions

P1′–P6, Section 6.6 of the paper. In short: the A3 gap must **widen** with
context; learned `R` must beat frozen random `R`, or the Hash Layers null holds
and the central claim is false; wall-clock gains appear only above a crossover
length, below which PHASOR is **slower** than dense FlashAttention.

## Citation

```bibtex
@techreport{doria2026phasor,
  title  = {PHASOR: Phase-Aware Sparse Attention with Learned Orthogonal Routing},
  author = {Doria, Alessio},
  year   = {2026},
  note   = {\url{https://github.com/Alessio2405/phasor}}
}
```
