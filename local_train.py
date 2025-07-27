# local_train.py
"""Prototype trainer — RTX 4060, 256x256
================================================
Trains **stage-1 VQ-VAE** and **stage-2 MatFormer** on a local
machine with limited 8 GB of VRAM.  Uses Hugging Face *accelerate*
so the exact same script can transparently scale to multi-GPU in
case you plug an eGPU later.

Usage
-----
$ accelerate config            # run once, choose "No distributed"
$ accelerate launch local_train.py \
      --data_root nasa_galaxy_images/resized \
      --epochs 25 --batch 4

The script writes.
  checkpoints/vqvae_*.pt   (best VQ-VAE)
  checkpoints/tfm_*.pt     (best MatFormer)
  wandb logs  (if WANDB_API_KEY set)
"""
from __future__ import annotations

import argparse, math, random, os, itertools
from pathlib import Path
from typing import Tuple

import torch, torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from accelerate import Accelerator
from modelTraining import build_vqvae, build_transformer, VQCfg, TFMCfg, maybe_autocast

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Local RTX‑4060 training – 256×256 prototype")
parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--batch", type=int, default=4)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--save_dir", type=str, default="checkpoints")
parser.add_argument("--log_wandb", action="store_true")
parser.add_argument("--resume", type=str, default=None)
args = parser.parse_args()
SAVE_DIR = Path(args.save_dir); SAVE_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Accelerator + logging
# -----------------------------------------------------------------------------
acc = Accelerator(fp16=True)  # mixed‑precision on CUDA
if acc.is_main_process and args.log_wandb:
    import wandb; wandb.init(project="machine-hallucinations-proto")

# -----------------------------------------------------------------------------
# Data pipeline — random 256² crops, flip
# -----------------------------------------------------------------------------
trans = T.Compose([
    T.RandomResizedCrop(256, scale=(0.9, 1.0)),
    T.RandomHorizontalFlip(),
    T.ToTensor()])
train_ds = ImageFolder(args.data_root, transform=trans)
train_dl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=4, pin_memory=True)

# -----------------------------------------------------------------------------
# Models + optimizers
# -----------------------------------------------------------------------------
vq = build_vqvae(VQCfg(codebook_size=1024, code_dim=256))
tfm = build_transformer(TFMCfg(vocab_size=1024, n_layers=6, d_model=512, n_heads=8, max_seq_len=256, grad_checkpoint=True))

opt_vq = torch.optim.AdamW(vq.parameters(), lr=args.lr, weight_decay=1e-4)
opt_tfm = torch.optim.AdamW(tfm.parameters(), lr=args.lr, weight_decay=1e-4)

# Prepare for Accelerate
vq, tfm, opt_vq, opt_tfm, train_dl = acc.prepare(vq, tfm, opt_vq, opt_tfm, train_dl)

# Optionally resume
if args.resume:
    ck = torch.load(args.resume, map_location="cpu")
    vq.load_state_dict(ck["vq"]); tfm.load_state_dict(ck["tfm"])

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
BETA_COMMIT = 0.25
for epoch in range(args.epochs):
    vq.train(); tfm.train()
    loss_vq_ema, loss_gen_ema = 0., 0.
    for step, (imgs, _) in enumerate(train_dl, 1):
        with maybe_autocast():
            recon, codes, commit_loss = vq(imgs)
            loss_recon = F.mse_loss(recon, imgs)
            loss_vq = loss_recon + commit_loss
        acc.backward(loss_vq)
        if step % 4 == 0:   # grad accum if batch small
            opt_vq.step(); opt_vq.zero_grad()
        loss_vq_ema = loss_vq_ema * 0.99 + loss_vq.item() * 0.01

        # ── Stage‑2 transformer training ───────────────────────────────
        seq = codes.view(codes.size(0), -1)          # (B, 16*16)
        logits = tfm(seq[:, :-1])
        loss_gen = F.cross_entropy(logits.view(-1, logits.size(-1)), seq[:, 1:].reshape(-1))
        acc.backward(loss_gen)
        if step % 4 == 0:
            opt_tfm.step(); opt_tfm.zero_grad()
        loss_gen_ema = loss_gen_ema * 0.99 + loss_gen.item() * 0.01

        if acc.is_main_process and step % 200 == 0:
            print(f"E{epoch} S{step} recon:{loss_recon.item():.4f} gen:{loss_gen.item():.4f}")
            if args.log_wandb:
                wandb.log({"loss_recon": loss_recon.item(), "loss_gen": loss_gen.item()})

    # Save epoch checkpoint (only main process to avoid race)
    if acc.is_main_process:
        ck_path = SAVE_DIR / f"epoch{epoch:03d}.pt"
        torch.save({"vq": acc.get_state_dict(vq), "tfm": acc.get_state_dict(tfm)}, ck_path)
        print("saved", ck_path)
