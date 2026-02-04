from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass(frozen=True)
class ECEConfig:
    n_bins: int = 15

def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y_true) ** 2))

def nll(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

def ece(y_true: np.ndarray, p: np.ndarray, cfg: ECEConfig = ECEConfig()) -> Tuple[float, Dict[str, np.ndarray]]:
    """Expected calibration error using confidence bins over p.

Returns (ece, bin_data) where bin_data contains:
- bin_edges, bin_acc, bin_conf, bin_count
"""
    p = np.asarray(p)
    y_true = np.asarray(y_true)
    bin_edges = np.linspace(0.0, 1.0, cfg.n_bins + 1)
    e = 0.0
    bin_acc = []
    bin_conf = []
    bin_count = []
    for i in range(cfg.n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (p >= lo) & (p < hi) if i < cfg.n_bins - 1 else (p >= lo) & (p <= hi)
        cnt = int(mask.sum())
        bin_count.append(cnt)
        if cnt == 0:
            bin_acc.append(np.nan)
            bin_conf.append(np.nan)
            continue
        acc_i = float((y_true[mask] == (p[mask] >= 0.5).astype(int)).mean())
        conf_i = float(p[mask].mean())
        bin_acc.append(acc_i)
        bin_conf.append(conf_i)
        e += (cnt / len(p)) * abs(acc_i - conf_i)
    return float(e), {
        "bin_edges": bin_edges,
        "bin_acc": np.array(bin_acc, dtype=float),
        "bin_conf": np.array(bin_conf, dtype=float),
        "bin_count": np.array(bin_count, dtype=int),
    }
