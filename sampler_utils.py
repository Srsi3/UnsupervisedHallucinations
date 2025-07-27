# sampler_utils.py
"""Sampling + interpolation helpers for MatFormer + VQ-VAE pipeline."""
from __future__ import annotations

import torch, math
from torch.nn import functional as F
from modelTraining import maybe_autocast

# -----------------------------------------------------------------------------
# Top‑k, top‑p samplers
# -----------------------------------------------------------------------------

def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    topk = torch.topk(logits, k)[0][..., -1, None]
    return torch.where(logits < topk, torch.full_like(logits, float("-inf")), logits)


def top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    sort = torch.sort(logits, descending=True)[0]
    cumsum = sort.softmax(-1).cumsum(-1)
    mask = cumsum - sort.softmax(-1) > p
    cut = mask.float().argmax(dim=-1, keepdim=True)
    thresh = sort.gather(-1, cut)
    return torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)

# -----------------------------------------------------------------------------
# Sequence generation (autoregressive)
# -----------------------------------------------------------------------------

def generate(model, start: torch.Tensor, steps: int, temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0, ratio: float | None = None) -> torch.Tensor:
    """Generate `steps` tokens after `start` sequence (B, L0)."""
    model.eval()
    seq = start
    device = next(model.parameters()).device
    with torch.no_grad(), maybe_autocast():
        for _ in range(steps):
            logits = model(seq[:, -model.max_seq_len :], ratio=ratio)[:, -1] / temperature
            if top_k > 0:
                logits = top_k_logits(logits, top_k)
            if top_p > 0:
                logits = top_p_logits(logits, top_p)
            probs = logits.softmax(-1)
            next_tok = torch.multinomial(probs, 1)
            seq = torch.cat([seq, next_tok], dim=1)
    return seq

# -----------------------------------------------------------------------------
# Latent SLERP interpolation (between two embedding grids)
# -----------------------------------------------------------------------------

def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between two latent tensors."""
    a_norm, b_norm = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    omega = (a_norm * b_norm).sum(-1).acos().clamp_min(1e-4)
    so = omega.sin()
    return (omega - t * omega).sin() / so * a + (t * omega).sin() / so * b


def interpolate_latents(lat1: torch.Tensor, lat2: torch.Tensor, n_frames: int):
    """Yield n_frames interpolated latents from lat1 → lat2 (inclusive)."""
    for i in range(n_frames):
        yield slerp(lat1, lat2, i / (n_frames - 1))
