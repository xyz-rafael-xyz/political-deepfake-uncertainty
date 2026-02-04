from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..models.backbones import enable_dropout

@dataclass(frozen=True)
class MCConfig:
    T: int = 20
    device: str = "cuda"

@torch.no_grad()
def mc_predict(model: torch.nn.Module, loader: DataLoader, cfg: MCConfig) -> Dict[str, np.ndarray]:
    """MC dropout predictions.

Returns dict with:
- ids: str array
- y: int array
- mu: mean p(y=1|x)
- var: predictive variance of p
- ent: predictive entropy of mu
- logits_det: single deterministic logits (dropout disabled)
"""
    model.eval()
    ids_all = []
    y_all = []

    # Collect T probability samples per batch
    mu_all = []
    var_all = []
    ent_all = []

    # Deterministic logits for reference
    logits_det_all = []

    for x, y, ids in loader:
        x = x.to(cfg.device)
        y = y.numpy().astype(int)
        ids = np.asarray(ids)

        # deterministic
        model.eval()
        logits_det = model(x).detach().cpu().numpy()

        # MC
        probs = []
        model.eval()
        enable_dropout(model)
        for _ in range(cfg.T):
            logits = model(x)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        probs = np.stack(probs, axis=0)  # [T, B]
        mu = probs.mean(axis=0)
        var = probs.var(axis=0)
        ent = -(mu * np.log(np.clip(mu, 1e-12, 1.0)) + (1 - mu) * np.log(np.clip(1 - mu, 1e-12, 1.0)))

        ids_all.append(ids)
        y_all.append(y)
        mu_all.append(mu)
        var_all.append(var)
        ent_all.append(ent)
        logits_det_all.append(logits_det)

    return {
        "ids": np.concatenate(ids_all),
        "y": np.concatenate(y_all),
        "mu": np.concatenate(mu_all),
        "var": np.concatenate(var_all),
        "ent": np.concatenate(ent_all),
        "logits_det": np.concatenate(logits_det_all),
    }
