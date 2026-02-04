#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from polideep.data.dataset import ImageCSVDataset
from polideep.models.backbones import BinaryClassifier, ModelConfig
from polideep.checkpoint import load_checkpoint
from polideep.train import predict_logits
from polideep.utils.device import get_default_device
from polideep.utils.io import ensure_dir, write_json

def build_transforms(res: int, imagenet_norm: bool):
    t = [transforms.Resize((res, res)), transforms.ToTensor()]
    if imagenet_norm:
        t.append(transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))
    return transforms.Compose(t)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/political_openfake")
    ap.add_argument("--split_csv", default="data/political_openfake/splits/test.csv")
    ap.add_argument("--backbone", choices=["resnet18","efficientnet_b4"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--resolution", type=int, default=380)
    ap.add_argument("--imagenet_norm", action="store_true")
    args = ap.parse_args()

    device = get_default_device()
    tfm = build_transforms(args.resolution, args.imagenet_norm)

    ds = ImageCSVDataset(args.data_root, args.split_csv, transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = BinaryClassifier(ModelConfig(backbone=args.backbone, pretrained=False))
    model = load_checkpoint(model, args.ckpt, device=device)

    logits, y, ids = predict_logits(model, loader, device=device)
    p = 1.0 / (1.0 + np.exp(-logits))

    out = ensure_dir(Path(args.out_dir) / f"{args.backbone}" / "predictions")
    np.save(out / "ids.npy", ids)
    np.save(out / "y.npy", y)
    np.save(out / "logits_det.npy", logits)
    np.save(out / "p_det.npy", p)
    write_json({"ckpt": args.ckpt, "resolution": args.resolution, "imagenet_norm": args.imagenet_norm}, out / "meta.json")
    print(f"Saved predictions to {out}")

if __name__ == "__main__":
    main()
