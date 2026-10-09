#!/usr/bin/env python3
"""E0 -- phase contamination in a pretrained RoPE model. No training required.

This is the first experiment in the protocol because it is cheap and because it
decides whether the paper's RoPE-decoupling contribution survives. It extracts
key vectors before and after RoPE from a pretrained model, clusters each on the
sphere, and asks how much of the cluster identity is explained by *position*
rather than by *content*.

Prediction (P1): post-RoPE clusters carry more positional and less lexical
information than pre-RoPE clusters, and the gap widens with context length.

Falsification: if pre- and post-RoPE are comparable, Proposition 2 has no
practical force. Say so in FINDINGS.md and withdraw the contribution. Do not
reformulate it.

Finding F1 (NOTES_FOR_PAPER.md) says the effective bucket count matters as much
as the mutual information, so we report both.

Usage:
    python run_e0.py --model EleutherAI/pythia-160m --contexts 512 1024 2048
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))


from e0_estimators import analyse  # noqa: E402


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


def extract_keys(model_name: str, text_batches, context: int, device: str):
    """Yield ``(layer, pre_rope_keys, post_rope_keys, token_ids, positions)``.

    Implemented against a Hugging Face model. Getting *pre*-RoPE keys requires
    hooking inside the attention module, and the hook point differs by
    architecture -- this is the part you will have to adapt, and the part worth
    checking twice: if you accidentally capture post-RoPE keys in both arms, E0
    trivially "fails to reproduce" the effect.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval().to(device)

    raise NotImplementedError(
        "Hook the attention modules of your chosen architecture here and return\n"
        "keys both before and after the rotary embedding is applied.\n\n"
        "For GPT-NeoX/Pythia the split lives in GPTNeoXAttention: the qkv tensor\n"
        "is divided into rotary and pass-through parts before\n"
        "`apply_rotary_pos_emb`. Capture the key tensor on both sides of that\n"
        "call. Verify the two differ -- if they do not, the hook is wrong and\n"
        "every number below is meaningless."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--contexts", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--clusters", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--n-sequences", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    results = []
    for context in args.contexts:
        for layer, pre, post, tokens, positions in extract_keys(
            args.model, None, context, args.device
        ):
            for B in args.clusters:
                for arm, keys in (("pre_rope", pre), ("post_rope", post)):
                    row = analyse(keys, positions, tokens, B)
                    row.update(model=args.model, context=context, layer=layer, arm=arm)
                    results.append(row)
                    print(json.dumps(row))

    pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {len(results)} rows to {args.out}")
    print("Now write FINDINGS.md. State whether P1 held. If it did not, say so.")


if __name__ == "__main__":
    main()
