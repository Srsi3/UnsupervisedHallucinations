# cloud_train.py
"""High-res (1024x1024) stage-2 training on A100 / 4090
========================================================
Assumes a **pre-trained VQ-VAE** checkpoint from local run
(or trained separately on crops).  Loads 1024x1024 images
(or random 1024 crops) and trains the MatFormer transformer
with sequence length ≤1024 (f = 32).

Usage (RunPod 40 GB A100):
$ python cloud_train.py --data_root nasa_galaxy_images/resized \
                       --vq_ckpt checkpoints/best_vqvae.pt \
                       --epochs 8 --batch 8

Uses DeepSpeed ZeRO-1 to shard optimizer state and
HuggingFace *accelerate* for launch/resume.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
from functools import partial

import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from accelerate import Accelerator
from modelTraining import build_vqvae, build_transformer, VQCfg, TFMCfg, maybe_autocast

parser = argparse.ArgumentParser(description="Cloud 1024 training")
parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--vq_ckpt", type=str, required=True)
parser.add_argument("--epochs", type=int, default=8)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--save_dir", type=str, default="cloud_ckpts")
args = parser.parse_args()
SAVE_DIR = Path(args.save_dir); SAVE_DIR.mkdir(exist_ok=True)

acc = Accelerator(fp16=True, split_batches=True, deepspeed_plugin={"zero_stage": 1})

# Data – random 1024 crops (we trained VQ‑VAE with f=32)
trans = T.Compose([
    T.RandomResizedCrop(1024, scale=(0.9, 1.0)),
    T.RandomHorizontalFlip(),
    T.ToTensor()])
train_ds = ImageFolder(args.data_root, transform=trans)
train_dl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=8, pin_memory=True)

# Load frozen VQ‑VAE
vq = build_vqvae(VQCfg(codebook_size=4096, code_dim=256))
ck = torch.load(args.vq_ckpt, map_location="cpu")
vq.load_state_dict(ck)
vq.eval(); vq.requires_grad_(False)

# Transformer config – big model
cfg_tfm = TFMCfg(vocab_size=4096, n_layers=24, d_model=1024, n_heads=16, max_seq_len=1024, grad_checkpoint=False)
tfm = build_transformer(cfg_tfm)
opt = torch.optim.AdamW(tfm.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-2)

vq, tfm, opt, train_dl = acc.prepare(vq, tfm, opt, train_dl)

for epoch in range(args.epochs):
    tfm.train()
    for step, (imgs, _) in enumerate(train_dl, 1):
        with maybe_autocast(dtype=torch.bfloat16):  # A100 bfloat16
            z_q, codes, _ = vq.encode(imgs)
            seq = codes.view(codes.size(0), -1)
            logits = tfm(seq[:, :-1])
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), seq[:, 1:].reshape(-1))
        acc.backward(loss)
        opt.step(); opt.zero_grad()

        if acc.is_main_process and step % 100 == 0:
            print(f"E{epoch} S{step} loss {loss.item():.4f}")

    if acc.is_main_process:
        path = SAVE_DIR / f"tfm_e{epoch:02d}.pt"
        torch.save(acc.get_state_dict(tfm), path)
        print("saved", path)
