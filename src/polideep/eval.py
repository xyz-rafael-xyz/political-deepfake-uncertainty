from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np

from .metrics.classification import accuracy, roc_auc, confusion
from .metrics.calibration import ece, ECEConfig, brier_score, nll

@dataclass(frozen=True)
class EvalConfig:
    threshold: float = 0.5
    ece_bins: int = 15

def evaluate_binary(y: np.ndarray, p: np.ndarray, cfg: EvalConfig) -> Dict[str, Any]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    yhat = (p >= cfg.threshold).astype(int)

    acc = accuracy(y, yhat)
    auc = roc_auc(y, p)
    ece_val, _ = ece(y, p, ECEConfig(n_bins=cfg.ece_bins))
    brier = brier_score(y, p)
    nll_val = nll(y, p)
    cm = confusion(y, p, cfg.threshold)

    return {
        "Acc": acc,
        "AUC": auc,
        "ECE": ece_val,
        "Brier": brier,
        "NLL": nll_val,
        **cm,
    }
