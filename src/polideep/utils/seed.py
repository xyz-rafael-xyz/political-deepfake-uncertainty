from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class SeedConfig:
    seed: int = 42
    deterministic: bool = True


def set_all_seeds(cfg: SeedConfig) -> None:
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    if torch is None:
        return

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
