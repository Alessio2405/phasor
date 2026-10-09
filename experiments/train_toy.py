#!/usr/bin/env python3
"""Toy-scale training harness for ablations A1/A3/A4 (Task 3 of the protocol).

Deliberately minimal. What it does guarantee:

- every arm is the same model with one attention flag changed;
- parameter counts, token counts and wall-clock are recorded per run, so a table
  can state what it is matched on instead of implying it;
- diagnostics from Section 5.4 (load entropy, summary mass, pattern density) are
  logged alongside loss, because perplexity alone cannot tell you whether the
  *router* worked.

What it does not do: produce publishable numbers. The attention reference is
`O(T^2)`, so wall-clock here says nothing about the method's speed. Use this to
check that an arm trains and to compare perplexity at matched FLOPs, and run the
real thing on a real harness.

Usage:
    python train_toy.py --variant phasor --steps 500 --context 512
    python train_toy.py --variant a3_route_on_rope --steps 500 --context 512
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from phasor.model import GPT, GPTConfig  # noqa: E402


def synthetic_batch(batch: int, context: int, vocab: int, device, generator):
    """A copying task with long-range structure.

    The second half of each sequence repeats the first half. A model that cannot
    reach back across the whole context cannot do better than unigram statistics
    on it, which makes this sensitive to exactly the thing sparse attention can
    break -- unlike random tokens, where any attention pattern scores the same.
    """
    half = context // 2
    prefix = torch.randint(0, vocab, (batch, half), generator=generator, device=device)
    seq = torch.cat([prefix, prefix], dim=1)[:, :context]
    return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()


def run(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    cfg = GPTConfig(
        vocab_size=args.vocab,
        n_layer=args.layers,
        n_head=args.heads,
        d_model=args.d_model,
        max_seq_len=args.context,
        attn_variant=args.variant,
        attn_kwargs=dict(window=args.window, rope_fraction=args.rope_fraction),
    )
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    n_params = model.n_params()
    tokens_per_step = args.batch * (args.context - 1)
    print(
        f"variant={args.variant}  non-embedding params={n_params:,}  "
        f"tokens/step={tokens_per_step:,}  device={device}"
    )

    history = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, y = synthetic_batch(args.batch, args.context, args.vocab, device, gen)
        opt.zero_grad(set_to_none=True)
        _, loss, stats = model(x, targets=y)
        (loss + stats["aux_loss"]).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "ppl": float(math.exp(min(float(loss.detach()), 20))),
                "aux_loss": float(stats["aux_loss"].detach()),
                "load_entropy": stats["load_entropy"],
                "summary_mass": stats["summary_mass"],
                "pattern_density": stats["pattern_density"],
                "tokens_seen": step * tokens_per_step,
                "elapsed_s": round(time.time() - t0, 2),
            }
            history.append(row)
            parts = [f"step {step:>5}", f"loss {row['loss']:.4f}", f"ppl {row['ppl']:8.2f}"]
            for key, label in (
                ("load_entropy", "H"),
                ("summary_mass", "sum"),
                ("pattern_density", "dens"),
            ):
                if row[key] is not None:
                    parts.append(f"{label} {row[key]:.3f}")
            print("  ".join(parts))

    out = {
        "config": vars(args),
        "n_params_non_embedding": n_params,
        "tokens_per_step": tokens_per_step,
        "total_tokens": args.steps * tokens_per_step,
        "wall_clock_s": round(time.time() - t0, 2),
        "history": history,
        "MATCHED_ON": "parameters and tokens; NOT FLOPs (reference impl is O(T^2))",
    }
    path = pathlib.Path(args.out or f"run_{args.variant}_seed{args.seed}.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")
    print(
        "Reminder: this harness cannot support a wall-clock claim. "
        "See NOTES_FOR_PAPER.md, 'Not implemented, and why it matters'."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="phasor")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--context", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d-model", dest="d_model", type=int, default=128)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--rope-fraction", dest="rope_fraction", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", dest="log_every", type=int, default=25)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
