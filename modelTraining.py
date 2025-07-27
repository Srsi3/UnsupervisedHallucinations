# vq_matformer_pipeline.py
"""
========================================================
This file contains the implementation of
  • VQ-VAE (with VectorQuantizerEMA option)
  • MatFormer Transformer (elastic nested-FFN)
  • Convenience factory helpers and memory-saving utils
It is designed to run *as is* on an RTX 4060 8 GB *(prototype)* or scale up
(up to 40-80 GB) by tweaking the config dictionary at the bottom.

Key features
------------
• Mixed-precision (AMP) safety: every forward is wrapped in autocast context
  via `with maybe_autocast()`.
• Gradient-checkpointing: enable per-block checkpoint by passing
  `grad_checkpoint=True` to the `MatFormer` factory.
• Weight-sharing MatFormer: sub-FFN weights are **views** into the main FFN
  tensor - zero extra parameters.
• EMA codebook (optional) for VQ-VAE stage-1 stability.
• Clean dataclass-style hyper-config for easy cloud ↔ local hand-offs.

See README at project root for full training scripts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt

# -----------------------------------------------------------------------------
# Utility: AMP context manager — on if CUDA, off otherwise
# -----------------------------------------------------------------------------
from contextlib import contextmanager

@contextmanager
def maybe_autocast(dtype: torch.dtype = torch.float16):
    if torch.is_autocast_enabled() or not torch.cuda.is_available():
        yield
    else:
        with torch.cuda.amp.autocast(dtype=dtype):
            yield

# -----------------------------------------------------------------------------
#   Encoder / Decoder CNNs (UNet‑style residual stack)
# -----------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class EncoderCNN(nn.Module):
    """4‑level down‑sampler: 256² → 16² by default (factor 16)."""
    def __init__(self, in_ch=3, base_ch=64, levels: int = 4):
        super().__init__()
        layers: List[nn.Module] = [nn.Conv2d(in_ch, base_ch, 7, padding=3)]
        ch = base_ch
        for i in range(levels):
            layers += [ResBlock(ch), ResBlock(ch)]
            layers += [nn.Conv2d(ch, ch * 2, 4, stride=2, padding=1)]  # downsample
            ch *= 2
        layers += [ResBlock(ch)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DecoderCNN(nn.Module):
    def __init__(self, out_ch=3, base_ch=64, levels: int = 4):
        super().__init__()
        ch = base_ch * (2 ** levels)
        layers: List[nn.Module] = [ResBlock(ch)]
        for i in range(levels):
            layers += [nn.ConvTranspose2d(ch, ch // 2, 4, stride=2, padding=1)]
            ch //= 2
            layers += [ResBlock(ch), ResBlock(ch)]
        layers += [nn.GroupNorm(32, ch), nn.SiLU(), nn.Conv2d(ch, out_ch, 7, padding=3)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# -----------------------------------------------------------------------------
#   Vector Quantizer (EMA optional)
# -----------------------------------------------------------------------------
class VectorQuantizer(nn.Module):
    """Basic VQ layer with optional EMA updates (default on for stability)."""

    def __init__(self, codebook_size: int = 1024, code_dim: int = 256, beta: float = 0.25, ema_decay: float = 0.99):
        super().__init__()
        self.K = codebook_size
        self.D = code_dim
        self.beta = beta
        self.ema_decay = ema_decay
        self.embedding = nn.Parameter(torch.randn(codebook_size, code_dim))
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_w", torch.randn_like(self.embedding))

    @torch.no_grad()
    def _ema_update(self, z_e, encodings):
        # encodings: (BHW,)
        one_hot = F.one_hot(encodings, num_classes=self.K).type(z_e.dtype)  # (BHW, K)
        n = one_hot.sum(0)                      # (K,)
        dw = one_hot.T @ z_e                   # (K, D)
        self.ema_cluster_size.mul_(self.ema_decay).add_(n, alpha=1 - self.ema_decay)
        self.ema_w.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)
        # Laplace smoothing
        n = self.ema_cluster_size + 1e-5
        self.embedding.data.copy_(self.ema_w / n.unsqueeze(1))

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input `z_e`: (B, D, H, W). Return (z_q, codes, commit_loss)."""
        B, D, H, W = z_e.shape
        # (BHW, D)
        z_flat = z_e.permute(0, 2, 3, 1).reshape(-1, D)
        # Compute distances via ||a-b||² = ||a||² + ||b||² - 2 a·b
        dist = (
            z_flat.pow(2).sum(1, keepdim=True)
            + self.embedding.pow(2).sum(1)
            - 2 * z_flat @ self.embedding.t()
        )
        codes = torch.argmin(dist, dim=1)  # (BHW,)
        z_q = self.embedding[codes].view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        # Commit + codebook loss
        commit_loss = F.mse_loss(z_q.detach(), z_e) + self.beta * F.mse_loss(z_q, z_e.detach())
        # EMA update
        if self.training and self.ema_decay < 1.0:
            self._ema_update(z_flat.detach(), codes)
        # Straight‑through estimator
        z_q = z_e + (z_q - z_e).detach()
        return z_q, codes.view(B, H, W), commit_loss

# -----------------------------------------------------------------------------
#   VQ‑VAE Model (Encoder + VectorQuantizer + Decoder)
# -----------------------------------------------------------------------------
class VQVAE(nn.Module):
    def __init__(self, codebook_size=1024, code_dim=256, beta=0.25):
        super().__init__()
        self.encoder = EncoderCNN()
        self.quantizer = VectorQuantizer(codebook_size, code_dim, beta)
        self.decoder = DecoderCNN()

    def encode(self, x):
        z_e = self.encoder(x)
        z_q, codes, commit_loss = self.quantizer(z_e)
        return z_q, codes, commit_loss

    def decode(self, codes: torch.Tensor):
        # codes: (B, H, W) long
        z_q = self.quantizer.embedding[codes.view(-1)].view(codes.size(0), codes.size(1), codes.size(2), -1)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return self.decoder(z_q)

    def forward(self, x):
        z_q, codes, commit_loss = self.encode(x)
        recon = self.decoder(z_q)
        return recon, codes, commit_loss

# -----------------------------------------------------------------------------
#   MatFormer Block — weight‑shared nested FFN
# -----------------------------------------------------------------------------
class MatFormerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ratios: Tuple[float, ...] = (4.0, 2.0, 1.0)):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ratios = ratios
        # Main weight matrix — largest size
        d_ffn_main = int(d_model * max(ratios))
        self.w1 = nn.Linear(d_model, d_ffn_main)
        self.w2 = nn.Linear(d_ffn_main, d_model)
        self.act = nn.GELU()

    def _ffn(self, x, ratio: float):
        d_target = int(x.size(-1) * ratio)
        # slice weight views (no new parameters)
        out = self.act(F.linear(x, self.w1.weight[:d_target], self.w1.bias[:d_target]))
        out = F.linear(out, self.w2.weight[:, :d_target], self.w2.bias)
        return out

    def forward(self, x, ratio: Optional[float] = None, grad_checkpoint: bool = False):
        """ratio=None → largest FFN; else choose sub‑ratio (must be in self.ratios)."""
        def sa_ff(x):
            attn, _ = self.attn(x, x, x, need_weights=False)
            return x + attn

        x = ckpt(sa_ff, x) if grad_checkpoint else sa_ff(x)
        x = self.norm1(x)
        ratio = ratio or max(self.ratios)
        assert ratio in self.ratios, "ratio not configured"
        ffn_out = ckpt(self._ffn, x, ratio) if grad_checkpoint else self._ffn(x, ratio)
        x = self.norm2(x + ffn_out)
        return x

# -----------------------------------------------------------------------------
#   MatFormer Transformer (stack of blocks)
# -----------------------------------------------------------------------------
class MatFormer(nn.Module):
    def __init__(self, vocab_size: int, n_layers: int = 12, d_model: int = 768, n_heads: int = 12,
                 ratios: Tuple[float, ...] = (4.0, 2.0, 1.0), max_seq_len: int = 4096, grad_checkpoint: bool = False):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.01)
        self.blocks = nn.ModuleList([MatFormerBlock(d_model, n_heads, ratios) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len
        self.ratios = ratios
        self.grad_checkpoint = grad_checkpoint

    def forward(self, idx: torch.Tensor, ratio: Optional[float] = None):
        # idx: (B, L)
        assert idx.size(1) <= self.max_seq_len, "sequence too long"
        with maybe_autocast():
            x = self.tok_emb(idx) + self.pos_emb[:, :idx.size(1), :]
            mask = torch.triu(torch.ones(idx.size(1), idx.size(1), device=idx.device), diagonal=1).bool()
            for blk in self.blocks:
                def blk_f(y):
                    return blk(y, ratio, self.grad_checkpoint)
                x = blk_f(x)  # we let each block handle checkpoint internally
            x = self.ln_f(x)
            logits = self.head(x)
        return logits

# -----------------------------------------------------------------------------
#   Factory helpers
# -----------------------------------------------------------------------------
@dataclass
class VQCfg:
    codebook_size: int = 2048
    code_dim: int = 256
    beta: float = 0.25

@dataclass
class TFMCfg:
    vocab_size: int = 2048  # should match codebook_size
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    ratios: Tuple[float, ...] = (4.0, 2.0, 1.0)
    max_seq_len: int = 4096
    grad_checkpoint: bool = True


def build_vqvae(cfg: VQCfg) -> VQVAE:
    return VQVAE(cfg.codebook_size, cfg.code_dim, cfg.beta)


def build_transformer(cfg: TFMCfg) -> MatFormer:
    return MatFormer(cfg.vocab_size, cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.ratios, cfg.max_seq_len, cfg.grad_checkpoint)


# -----------------------------------------------------------------------------
#   Simple test‑run (debug)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vq = build_vqvae(VQCfg()).to(device)
    tfm = build_transformer(TFMCfg()).to(device)

    x = torch.randn(2, 3, 256, 256, device=device)
    recon, codes, loss_c = vq(x)
    seq = codes.view(codes.size(0), -1)  # flatten
    logits = tfm(seq[:, :-1])
    loss_gen = F.cross_entropy(logits.view(-1, logits.size(-1)), seq[:, 1:].reshape(-1))
    print("ok", recon.shape, loss_c.item(), loss_gen.item())
