"""A minimal GPT for the toy-scale ablations (Task 3 of the protocol).

Deliberately small and boring: the only thing that varies between ablation arms
is the attention variant. Everything else -- MLP, norms, init, tying -- is held
fixed so that A1/A3/A4 measure what they claim to measure.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import make_attention


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    mlp_ratio: int = 4
    dropout: float = 0.0
    bias: bool = False
    max_seq_len: int = 4096
    attn_variant: str = "phasor"
    attn_kwargs: dict = None

    def __post_init__(self):
        if self.attn_kwargs is None:
            self.attn_kwargs = {}


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = cfg.mlp_ratio * cfg.d_model
        self.fc = nn.Linear(cfg.d_model, hidden, bias=cfg.bias)
        self.proj = nn.Linear(hidden, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.attn = make_attention(
            cfg.attn_variant,
            cfg.d_model,
            cfg.n_head,
            dropout=cfg.dropout,
            bias=cfg.bias,
            max_seq_len=cfg.max_seq_len,
            **cfg.attn_kwargs,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, collect: list | None = None):
        # The residual path is here, not inside the attention module. Under the
        # pure-skip ablation this is the *only* thing carrying an unrouted token
        # forward -- which is exactly the point A4 tests.
        h, aux = self.attn(self.ln1(x), return_aux=True)
        if collect is not None:
            collect.append(aux)
        x = x + h
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # tied
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(self, idx, targets=None):
        x = self.drop(self.tok_emb(idx))
        auxes: list = []
        for blk in self.blocks:
            x = blk(x, collect=auxes)
        logits = self.head(self.ln_f(x))

        loss = None
        aux_loss = torch.zeros((), device=idx.device)
        for a in auxes:
            if a.balance_loss is not None:
                aux_loss = aux_loss + a.balance_loss
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
        stats = {
            "aux_loss": aux_loss,
            "load_entropy": _mean([a.load_entropy for a in auxes]),
            "summary_mass": _mean([a.summary_mass for a in auxes]),
            "pattern_density": _mean([a.pattern_density for a in auxes]),
        }
        return logits, loss, stats


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None
