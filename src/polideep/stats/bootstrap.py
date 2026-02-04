from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

@dataclass(frozen=True)
class BootstrapConfig:
    n_resamples: int = 1000
    ci: float = 0.95
    seed: int = 42

def bootstrap_ci(metric_fn: Callable[[np.ndarray], float], values: np.ndarray, cfg: BootstrapConfig) -> Tuple[float, float, float]:
    """Bootstrap CI for a scalar metric computed from a 1D array of per-example values.

metric_fn: maps resampled array -> scalar
values: 1D
Returns: (mean_metric, lo, hi)
"""
    rng = np.random.default_rng(cfg.seed)
    n = len(values)
    stats = []
    for _ in range(cfg.n_resamples):
        idx = rng.integers(0, n, size=n)
        stats.append(metric_fn(values[idx]))
    stats = np.asarray(stats, dtype=float)
    mean = float(np.mean(stats))
    alpha = (1.0 - cfg.ci) / 2.0
    lo = float(np.quantile(stats, alpha))
    hi = float(np.quantile(stats, 1.0 - alpha))
    return mean, lo, hi
