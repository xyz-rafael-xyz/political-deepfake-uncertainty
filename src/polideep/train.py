from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics.calibration import nll
from .utils.io import ensure_dir

@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr_backbone: float = 1e-5
    lr_head: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 5
    num_workers: int = 4
    device: str = "cuda"

def _split_params(model: torch.nn.Module):
    # heuristic: head parameters contain 'head'
    head, backbone = [], []
    for n, p in model.named_parameters():
        (head if "head" in n else backbone).append(p)
    return backbone, head

@torch.no_grad()
def predict_logits(model: torch.nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_y, all_ids = [], [], []
    for x, y, ids in loader:
        x = x.to(device)
        logits = model(x).detach().cpu().numpy()
        all_logits.append(logits)
        all_y.append(y.numpy())
        all_ids.extend(list(ids))
    return np.concatenate(all_logits), np.concatenate(all_y), np.array(all_ids)

def train_model(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, out_dir: str | Path, cfg: TrainConfig) -> Dict[str, Any]:
    out = ensure_dir(out_dir)
    model = model.to(cfg.device)

    backbone_params, head_params = _split_params(model)
    opt = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": cfg.lr_backbone},
            {"params": head_params, "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay,
    )

    criterion = nn.BCEWithLogitsLoss()
    best_val = float("inf")
    best_epoch = -1
    bad = 0

    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.epochs}")
        for x, y, _ids in pbar:
            x = x.to(cfg.device)
            y = y.float().to(cfg.device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.detach().cpu().item()))

        # Validation
        logits_v, y_v, _ = predict_logits(model, val_loader, cfg.device)
        p_v = 1.0 / (1.0 + np.exp(-logits_v))
        val_nll = nll(y_v, p_v)
        if val_nll < best_val - 1e-6:
            best_val = val_nll
            best_epoch = epoch
            bad = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_nll": best_val}, out / "best.pt")
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    return {"best_epoch": best_epoch, "best_val_nll": best_val, "checkpoint": str(out / "best.pt")}
