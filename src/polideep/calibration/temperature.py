from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

@dataclass
class TemperatureScaler(nn.Module):
    """Scalar temperature scaling: p = sigmoid(logit / tau)."""
    init_log_tau: float = 0.0  # tau = exp(log_tau)

    def __post_init__(self):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(self.init_log_tau, dtype=torch.float32))

    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.tau

def fit_temperature(logits: np.ndarray, y: np.ndarray, max_iter: int = 500, lr: float = 0.05, device: str = "cpu") -> float:
    """Fit scalar temperature on validation logits by minimizing NLL."""
    scaler = TemperatureScaler().to(device)
    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss()
    opt = optim.LBFGS([scaler.log_tau], lr=lr, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = criterion(scaler(logits_t), y_t)
        loss.backward()
        return loss

    opt.step(closure)
    tau = float(torch.exp(scaler.log_tau).detach().cpu().item())
    return tau

def apply_temperature(logits: np.ndarray, tau: float) -> np.ndarray:
    return logits / float(tau)
