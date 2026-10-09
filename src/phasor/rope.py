"""Rotary position embeddings, with support for partial application.

PHASOR splits each head into a positional part (``d_p`` dims, RoPE applied) and a
content part (``d_c`` dims, untouched). The router reads only the content part.
See Proposition 2 of the paper for why.
"""

from __future__ import annotations

import torch


def build_rope_cache(
    seq_len: int,
    dim: int,
    base: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(cos, sin)`` of shape ``(seq_len, dim)`` for a RoPE subspace.

    ``dim`` must be even; it is the size of the *positional* subspace, not the
    full head dimension.
    """
    if dim % 2 != 0:
        raise ValueError(f"RoPE dimension must be even, got {dim}")
    if dim == 0:
        empty = torch.zeros(seq_len, 0, device=device, dtype=dtype)
        return empty, empty

    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (seq_len, half)
    emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rope_frequencies(dim: int, base: float = 10000.0) -> torch.Tensor:
    """The per-plane angular frequencies ``theta_j``, for frequency-tercile analysis."""
    half = dim // 2
    return 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to ``x`` of shape ``(..., T, dim)``.

    ``cos``/``sin`` are ``(T, dim)`` and broadcast over the leading dims.
    """
    if x.shape[-1] == 0:
        return x
    T = x.shape[-2]
    cos = cos[:T].to(x.dtype)
    sin = sin[:T].to(x.dtype)
    return x * cos + _rotate_half(x) * sin


def split_head(x: torch.Tensor, d_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``(..., d_head)`` into ``(positional, content)``."""
    return x[..., :d_pos], x[..., d_pos:]


def apply_partial_rope(
    x: torch.Tensor,
    d_pos: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(x_rotated_full, x_content)``.

    ``x_rotated_full`` is the concatenation ``[RoPE(x_p) ; x_c]`` used for
    attention scores. ``x_content`` is ``x_c`` alone, which is what the router
    reads.
    """
    x_p, x_c = split_head(x, d_pos)
    x_p = apply_rope(x_p, cos, sin)
    return torch.cat([x_p, x_c], dim=-1), x_c
