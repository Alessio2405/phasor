# Notes for the paper

Things writing and testing the reference implementation forced into the open.
Some are findings that change what the paper should claim; some are design
decisions the paper does not specify and a reader could not reconstruct. Both
kinds belong in the next revision.

Everything under "Findings" was produced by code in this repo and is reproducible
with `pytest tests/`. Nothing here is a training result — no model has been
trained.

---

> **Status, October 2026.** F1--F4 below have been folded into the paper and are
> now backed by the controlled numerical studies in `experiments/numerical_studies.py`
> (Section 5 of the paper). Two further findings emerged from running those
> studies at scale: **F5** (chunk fidelity) and **F6** (the initialisation null).
> Regenerate every number with:
>
> ```
> python experiments/numerical_studies.py && python experiments/make_tables.py
> ```

## Findings

### F1 — Proposition 2's asymptotic `1/B` overstates the aggregate effect

**What the paper said.** A router reading a RoPE plane of frequency θ co-routes
content-identical tokens at chance rate `1/B₂` as the phase equidistributes.

**What the full sweep shows** (study S1/S2, `d_c` from 8 to 64, context to 32k):

| `d_c` | `B` | agreement @32k | chance | ratio | effective buckets | % of codebook |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 0.190 | 0.063 | 3.0× | 5.91 | 37% |
| 16 | 32 | 0.199 | 0.031 | 6.4× | 6.69 | 21% |
| 32 | 64 | 0.141 | 0.016 | 9.0× | 8.92 | 14% |
| 64 | 128 | 0.142 | 0.008 | 18.2× | 10.15 | **8%** |

The plateau is width-independent in absolute terms, so the *ratio* to chance
grows with `B`. The router reaches 5–10 effective buckets however many exist.

**Why — this is now Proposition 4 in the paper.** RoPE rotates within each 2-plane,
so plane energies are invariant and the trajectory over positions lies on a torus
of dimension `d_c/2` — a measure-zero subset of the sphere. Most of the partition
is unreachable by construction.

**Consequence.** Provisioning a bigger codebook makes the relative collapse
*worse*. The paper's corollary has been replaced.

### F2 — The load is a property of the data, not of the router

Now **Proposition 3** in the paper, with the full sweep in study S3:

| concentration κ | load entropy at init | after 300 steps of collapse pressure | Δ |
|---:|---:|---:|---:|
| 0.0 | 0.9994 | 0.9994 | −0.0000 |
| 1.0 | 0.9473 | 0.9476 | +0.0003 |
| 2.0 | 0.7151 | 0.7059 | −0.0092 |
| 4.0 | 0.1024 | 0.0787 | −0.0237 |

The load is a clean monotone function of input anisotropy; the router's
parameters move it by at most 0.024 anywhere, and by nothing at all when the
input is isotropic.

**Consequence: ablation A6 as specified could not work** and has been replaced in
the paper with A6′, which tests encoder-side balancing and routing-input
whitening instead.

### F3 — The residue branch swallows 62% of the attention mass at zero bias

Study S6, measured across bias initialisations:

| bias at init | residue mass | inside P3's 2–25% band |
|---:|---:|:--|
| 0.0 | 0.618 | no |
| −2.0 | 0.188 | yes |
| −4.0 | 0.031 | yes |
| −6.0 | 0.004 | no |

The branch's share spans two orders of magnitude over a hyperparameter the paper
did not declare, which made P3 unfalsifiable. The implementation now defaults to
−4; P3′ is stated for the converged model and the bias is swept in A4.

### F4 — A transpose bug that every orthogonality test would have passed

`HouseholderRotation.rotate` applied its reflections in forward order, giving
`x @ R` where the base class contract is `x @ Rᵀ`. Both are perfectly orthogonal
maps, so every orthogonality and norm-preservation check passed; only
`test_rotate_matches_matrix` caught it. The shape of bug this area invites: a
wrong answer that satisfies every invariant you thought to check.

### F5 — Chunking is faithful only when the chunk matches the bucket (new)

Study S7. The `O(n^1.5)` complexity depends on bucket-sorted fixed-size chunks,
which is **not the same function** as bucket-wise attention:

| context | chunk | chunk / mean bucket | load entropy | in-bucket recall |
|---:|---:|---:|---:|---:|
| 4096 | 32 | 0.25 | 0.996 | **0.378** |
| 4096 | 64 | 0.50 | 0.996 | 0.693 |
| 4096 | 128 | 1.00 | 0.996 | 0.927 |

Load entropy is near-perfect, so this is not an imbalance artefact — it is
intrinsic. Plotted against `c/m̄` the three context lengths collapse onto one
curve, identifying the governing variable.

**Consequence.** `c ≳ n/B` is a correctness constraint, so the in-bucket term is
`Ω(n²/B)` and `O(n^1.5)` holds only if `B ∝ √n`. Since `B = 2·d_c`, the routing
width must scale as `√n/2`: `d_c = 32` at n=4096, `d_c = 128` at n=65536. The
paper now states this as the design's most consequential limitation, and
decoupling `B` from `d_c` is the first change worth trying.

### F6 — At initialisation every ablation arm is identical (new)

Study S5. Recall of dense top-k mass at matched density, across all four arms and
both input regimes (Gaussian and content-structured):

- cross-polytope bucketing beats a random pattern of equal density by **2.5–4.7×**,
  and the margin grows with context;
- but the maximum spread *between arms* in any matched cell is **0.0101**.

Learned vs frozen rotation, content vs rotary routing, summaries vs skip — all
indistinguishable before training, even with content structure planted in the
input.

**Consequence.** A1 and A3 cannot be settled by pattern-quality measurement and
require full training. It also means an initialisation-time pattern comparison —
ours or anyone's — says nothing about routing rules.

### F7 — The training budget decides the answer (new)

On the copying task at toy scale, final losses read: dense 3.434 / 3.870,
PHASOR 4.158 / 4.160 (the unigram floor). Read as a result that says the sparse
arm cannot do the task. **The trajectories say otherwise.**

```
dense  seed0: 4.192 4.166 4.166 4.161 4.160 4.159 4.163 4.162 4.163 4.161 3.870
dense  seed1: 4.189 4.166 4.165 4.158 3.860 3.844 3.738 3.682 3.617 3.499 3.434
phasor seed0: 4.195 4.169 4.167 4.161 4.164 4.160 4.160 4.160 4.160 4.160 4.160
phasor seed1: 4.184 4.169 4.159 4.164 4.163 4.164 4.163 4.163 4.158 4.162 4.161
```

The task is learned by a **phase transition** whose onset varies by more than 2×
across seeds of an identical configuration: dense seed 1 leaves the floor at step
400, dense seed 0 at step 900 and is still mid-transition when the budget ends.
PHASOR's two seeds are both still flat.

So: within a budget where dense barely completes its transition, PHASOR has not
begun one. Whether it would at step 2000 is unmeasured, and nothing here
licenses "PHASOR cannot do it".

**The trap.** If the budget is set so the dense baseline is *near* its
transition rather than well past it, every slower-onset arm reports at the floor
and the comparison looks decisive when it is only truncated. Budgets for
A1/A3/A4 must run until the dense arm has **plateaued**, and trajectories — not
final numbers — must be reported.

(An earlier draft of this note read the 0.44 gap between dense seeds as
steady-state variance. It is not; it is onset timing. Corrected here.)

---

## Underspecified in the paper

### U1 — What space do the residue summaries live in?

The paper writes `k̄_b = mean(k_j)` without saying whether the keys are pre- or
post-RoPE. It matters: averaging post-RoPE keys averages vectors at different
phases, so the mean suffers destructive interference and shrinks toward zero as
the bucket grows — the summary degrades exactly when it carries the most tokens.

The implementation defaults to `summary_space="content"` (pool the RoPE-free
part, score against `q_c`), with `"full"` available as ablation arm
`a4_summaries_full_space`. Values are always pooled whole, since they carry no
positional phase.

Consequence: the residue branch has no positional information at all. That is
defensible — it is a coarse global branch, like a register or an attention sink —
but it is a design decision, not a detail, and the paper should state it.

### U2 — The two branches score over different subspaces inside one softmax

Fine logits are `q·k` over `d_head` dims scaled by `1/√d_head`; residue logits
are `q_c·k̄_c` over `d_c` dims. Putting them in one softmax with the same scale
is not obviously right. The implementation adds a learnable per-head scale and
bias on the residue branch so the model can calibrate it. The paper's Section 3.4
should say this, and P3's "2–25% summary mass" prediction is partly a prediction
about whether that calibration converges somewhere sensible.

### U3 — Which round's buckets define the summaries?

With `n_rounds > 1` a key belongs to a different bucket per round. Maintaining
summaries per round multiplies memory by `n_r`. The implementation uses round 0
only. Arbitrary; needs either a justification or an ablation.

### U4 — Window and summaries double-count

A key inside the sliding window that belongs to a non-selected bucket is attended
finely *and* contributes to that bucket's summary. Small effect, but it means the
residue branch is not exactly "the complement of what the router selected", which
is the phrase the paper uses to distinguish it from NSA's compression branch.
Either fix the phrase or subtract the window from the summaries.

### U5 — The balancing loss reads future tokens

`L_bal` is computed over all positions in the batch, including positions after
the current one. This does not break autoregression — it is a batch-level
regulariser on parameters, exactly like standard MoE auxiliary losses, and
`tests/test_causality.py` confirms the model *function* stays causal. Worth one
sentence so a reviewer does not have to work it out.

---

## Not implemented, and why it matters

**The chunked gather path does not exist** (and F5 now quantifies what that costs). `PhasorAttention` builds the full
`(T, T)` score matrix and applies routing as a mask. It is `O(T²)` in time and
memory. Everything about *semantics* — recall, load entropy, summary mass,
perplexity — is measurable with it; nothing about *speed* is.

The paper's `O(T^1.5)` in Section 4.1 assumes bucket-sorted fixed-size chunks à
à la Reformer. That introduces a problem the paper only treats asymptotically:
**chunking with unbalanced buckets is not the same function as bucket-wise
attention.** A bucket larger than the chunk gets split across chunks and loses
intra-bucket pairs; a bucket smaller than the chunk gets padded with tokens from
a neighbouring bucket. So the chunked implementation computes a *different*
attention pattern from the one the method describes, and the size of that gap is
an empirical question nobody has answered here.

Given F2 — that load balance depends on encoder anisotropy and not on the router
— this gap could be large. It should be measured before any wall-clock claim, and
the paper should state the chunked pattern as the actual method rather than
presenting chunking as an implementation detail of the ideal one.
