from __future__ import annotations

from pathlib import Path
import torch

def load_checkpoint(model: torch.nn.Module, ckpt_path: str | Path, device: str = "cpu") -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    return model
