#!/usr/bin/env python3
"""Generate the paper's LaTeX tables and figures from the study CSVs.

Every number in the paper's results section comes through here. Nothing is typed
by hand, so the tables cannot drift from the runs that produced them, and
re-running `numerical_studies.py` followed by this script regenerates the paper's
evidence end to end.

Writes into ../paper/generated/.
"""

from __future__ import annotations

import csv
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "results"
OUT = HERE.parent / "paper" / "generated"
OUT.mkdir(parents=True, exist_ok=True)


def load(name: str) -> list[dict]:
    path = RESULTS / f"{name}.csv"
    if not path.exists():
        print(f"  !! missing {path.name}; skipping")
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def emit(name: str, body: str) -> None:
    (OUT / f"{name}.tex").write_text(body)
    print(f"  -> generated/{name}.tex")


def tabular(header: list[str], rows: list[list[str]], align: str) -> str:
    out = [f"\\begin{{tabular}}{{{align}}}", "\\toprule"]
    out.append(" & ".join(header) + " \\\\")
    out.append("\\midrule")
    out.extend(" & ".join(r) + " \\\\" for r in rows)
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


# ---------------------------------------------------------------------------


def table_s1_s2() -> None:
    s1 = {(r["d_c"], r["context"]): r for r in load("s1_phase_contamination")}
    s2 = {(r["d_c"], r["context"]): r for r in load("s2_effective_codebook")}
    if not s1:
        return
    contexts = ["512", "2048", "8192", "32768"]
    rows = []
    for d_c in ("8", "16", "32", "64"):
        agree = [s1[(d_c, c)]["agreement"] for c in contexts]
        eff = [s2[(d_c, c)]["effective_buckets"] for c in contexts if (d_c, c) in s2]
        B = s1[(d_c, contexts[0])]["n_buckets"]
        rows.append([d_c, B] + agree + eff)
    header = (
        ["$\\dc$", "$B$"]
        + [f"\\multicolumn{{1}}{{c}}{{{c}}}" for c in contexts]
        + [f"\\multicolumn{{1}}{{c}}{{{c}}}" for c in contexts]
    )
    body = tabular(header, rows, "ll" + "r" * 8)
    body = body.replace(
        "\\toprule",
        "\\toprule\n & & \\multicolumn{4}{c}{co-routing agreement}"
        " & \\multicolumn{4}{c}{effective buckets} \\\\\n"
        "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}",
    )
    emit("tab_s1s2", body)


def table_s3() -> None:
    rows = load("s3_load_vs_anisotropy")
    if not rows:
        return
    body = tabular(
        ["$\\kappa$", "$H(\\pi)/\\log B$ at init", "after pressure", "$\\Delta$", "buckets used"],
        [
            [
                r["concentration"],
                r["load_entropy_init"],
                r["load_entropy_after_pressure"],
                r["delta"],
                r["buckets_used"],
            ]
            for r in rows
        ],
        "lrrrr",
    )
    emit("tab_s3", body)


def table_s4() -> None:
    rows = load("s4_codebook_geometry")
    if not rows:
        return
    steps = sorted({int(r["step"]) for r in rows})
    by = {}
    for r in rows:
        by.setdefault(r["router"], {})[int(r["step"])] = r["min_prototype_angle_deg"]
    body = tabular(
        ["router"] + [f"step {s}" for s in steps],
        [[name] + [by[name].get(s, "--") for s in steps] for name in by],
        "l" + "r" * len(steps),
    )
    emit("tab_s4", body)


def table_s5() -> None:
    rows = load("s5_pattern_quality")
    if not rows:
        return
    sel = [r for r in rows if r["window"] == "8"]
    body = tabular(
        ["input", "variant", "$n$", "density", "top-$k$ recall", "recall/density"],
        [
            [
                r["regime"],
                r["variant"].replace("_", "\\_"),
                r["context"],
                r["density"],
                r["topk_recall"],
                r["recall_over_density"],
            ]
            for r in sel
        ],
        "llrrrr",
    )
    emit("tab_s5", body)


def table_s6() -> None:
    rows = load("s6_residue_calibration")
    if not rows:
        return
    body = tabular(
        ["bias at init", "residue mass", "inside P3's 2--25\\% band"],
        [
            [r["summary_bias_init"], r["summary_mass"], "yes" if r["within_P3_band"] == "True" else "no"]
            for r in rows
        ],
        "lrl",
    )
    emit("tab_s6", body)


def table_s7() -> None:
    rows = load("s7_chunk_divergence")
    if not rows:
        return
    out = []
    for r in rows:
        T, c = int(r["context"]), int(r["chunk_size"])
        avg_bucket = T / 32.0  # B = 2*d_c = 32 at d_model 128, 4 heads
        out.append(
            [
                str(T),
                str(c),
                f"{c / avg_bucket:.2f}",
                r["load_entropy"],
                r["inbucket_recall"],
                r["contamination"],
            ]
        )
    body = tabular(
        [
            "$n$",
            "chunk $c$",
            "$c / \\bar{m}$",
            "$H(\\pi)/\\log B$",
            "in-bucket recall",
            "contamination",
        ],
        out,
        "rrrrrr",
    )
    emit("tab_s7", body)


def table_ablation() -> None:
    """Toy-scale ablation arms, if the sweep has produced runs."""
    import json

    files = sorted(RESULTS.glob("*_seed*.json"))
    if not files:
        print("  !! no ablation runs yet; skipping tab_ablation")
        return
    by_arm: dict[str, dict[int, dict]] = {}
    for f in files:
        data = json.loads(f.read_text())
        arm = data["config"]["variant"]
        seed = data["config"]["seed"]
        hist = data["history"]
        final = hist[-1]
        best = min(h["loss"] for h in hist)
        by_arm.setdefault(arm, {})[seed] = dict(
            final_loss=final["loss"],
            best_loss=best,
            load_entropy=final["load_entropy"],
            summary_mass=final["summary_mass"],
            density=final["pattern_density"],
            params=data["n_params_non_embedding"],
            tokens=data["total_tokens"],
            wall=data["wall_clock_s"],
        )

    order = ["dense", "phasor", "a1_frozen_rotation", "a3_route_on_rope", "a4_pure_skip"]
    rows = []
    for arm in order:
        if arm not in by_arm:
            continue
        seeds = sorted(by_arm[arm])
        losses = [f"{by_arm[arm][s]['best_loss']:.3f}" for s in seeds]
        any_run = by_arm[arm][seeds[0]]
        dens = any_run["density"]
        ent = any_run["load_entropy"]
        rows.append(
            [
                arm.replace("_", "\\_"),
                f"{any_run['params']:,}".replace(",", "\\,"),
                " / ".join(losses),
                f"{dens:.3f}" if dens is not None else "--",
                f"{ent:.3f}" if ent is not None else "--",
            ]
        )
    body = tabular(
        ["arm", "params", "best loss (seed 0 / 1)", "density", "$H(\\pi)/\\log B$"],
        rows,
        "lrrrr",
    )
    emit("tab_ablation", body)


# ---------------------------------------------------------------------------


def figures() -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  !! matplotlib missing; skipping figures")
        return

    plt.rcParams.update(
        {
            "font.size": 8,
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )
    BLUE, RED, GREY = "#0050A0", "#B03030", "#808080"

    s1 = load("s1_phase_contamination")
    s2 = load("s2_effective_codebook")
    s7 = load("s7_chunk_divergence")
    if not (s1 and s2 and s7):
        print("  !! missing study data; skipping figures")
        return

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9))

    # (a) agreement vs context
    ax = axes[0]
    for i, d_c in enumerate(("8", "16", "32", "64")):
        pts = [(int(r["context"]), float(r["agreement"])) for r in s1 if r["d_c"] == d_c]
        pts.sort()
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            marker="o",
            ms=3,
            lw=1.2,
            color=plt.cm.viridis(i / 3.5),
            label=f"$d_c={d_c}$",
        )
    for i, d_c in enumerate(("8", "16", "32", "64")):
        B = int([r for r in s1 if r["d_c"] == d_c][0]["n_buckets"])
        ax.axhline(1.0 / B, color=GREY, lw=0.5, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("context $n$")
    ax.set_ylabel("co-routing agreement")
    ax.set_title("(a) phase contamination", fontsize=8)
    ax.legend(frameon=False, fontsize=6.5)

    # (b) effective codebook as a fraction of B
    ax = axes[1]
    for i, d_c in enumerate(("8", "16", "32", "64")):
        pts = [
            (int(r["context"]), float(r["effective_buckets"]))
            for r in s2
            if r["d_c"] == d_c
        ]
        pts.sort()
        B = int([r for r in s2 if r["d_c"] == d_c][0]["n_buckets"])
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            marker="o",
            ms=3,
            lw=1.2,
            color=plt.cm.viridis(i / 3.5),
            label=f"$B={B}$",
        )
        ax.axhline(B, color=plt.cm.viridis(i / 3.5), lw=0.5, ls="--", alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("context $n$")
    ax.set_ylabel("effective buckets")
    ax.set_title("(b) codebook capacity ceiling", fontsize=8)
    ax.legend(frameon=False, fontsize=6.5)

    # (c) chunk fidelity vs chunk/bucket ratio
    ax = axes[2]
    for i, T in enumerate(("256", "1024", "4096")):
        pts = [
            (int(r["chunk_size"]) / (int(T) / 32.0), float(r["inbucket_recall"]))
            for r in s7
            if r["context"] == T
        ]
        pts.sort()
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            marker="s",
            ms=3,
            lw=1.2,
            color=plt.cm.plasma(i / 3.5),
            label=f"$n={T}$",
        )
    ax.axhline(1.0, color=GREY, lw=0.5, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"chunk size / mean bucket size")
    ax.set_ylabel("in-bucket recall")
    ax.set_title("(c) chunking fidelity", fontsize=8)
    ax.legend(frameon=False, fontsize=6.5)

    fig.tight_layout()
    fig.savefig(OUT / "studies.pdf", bbox_inches="tight")
    print("  -> generated/studies.pdf")


def main():
    print("generating tables and figures")
    table_s1_s2()
    table_s3()
    table_s4()
    table_s5()
    table_s6()
    table_s7()
    table_ablation()
    figures()
    print("done")


if __name__ == "__main__":
    main()
