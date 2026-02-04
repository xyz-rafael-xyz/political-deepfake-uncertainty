from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import timm

BackboneName = Literal["resnet18", "efficientnet_b4"]

@dataclass(frozen=True)
class ModelConfig:
    backbone: BackboneName
    pretrained: bool = True
    dropout_p: float = 0.0  # optional extra dropout before classifier

class BinaryClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.net = timm.create_model(cfg.backbone, pretrained=cfg.pretrained, num_classes=0, global_pool="avg")
        feat_dim = self.net.num_features
        self.head = nn.Sequential(
            nn.Dropout(p=cfg.dropout_p) if cfg.dropout_p > 0 else nn.Identity(),
            nn.Linear(feat_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.net(x)
        z = self.head(f).squeeze(1)
        return z

def enable_dropout(model: nn.Module) -> None:
    """Enable dropout modules during evaluation (MC dropout)."""
    for m in model.modules():
        if isinstance(m, nn.Dropout) or m.__class__.__name__.lower().startswith("dropout"):
            m.train()
