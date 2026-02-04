#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from polideep.data.dataset import ImageCSVDataset
from polideep.models.backbones import BinaryClassifier, ModelConfig
from polideep.checkpoint import load_checkpoint
from polideep.uncertainty.mc_dropout import mc_predict, MCConfig
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
    ap.add_argument("--split_csv", required=True)
    ap.add_argument("--backbone", choices=["resnet18","efficientnet_b4"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--T", type=int, default=20)
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

    out = ensure_dir(Path(args.out_dir) / f"{args.backbone}" / "predictions")
    res = mc_predict(model, loader, MCConfig(T=args.T, device=device))

    # Save
    np.save(out / f"p_mc_mean_T{args.T}.npy", res["mu"])
    np.save(out / f"var_mc_T{args.T}.npy", res["var"])
    np.save(out / f"ent_mc_T{args.T}.npy", res["ent"])

    # Also save a single-pass sample as control (T=1)
    if args.T != 1:
        res1 = mc_predict(model, loader, MCConfig(T=1, device=device))
        np.save(out / "p_mc_T1.npy", res1["mu"])

    write_json({"T": args.T, "ckpt": args.ckpt}, out / f"mc_T{args.T}.json")
    print(f"Saved MC outputs to {out}")

if __name__ == "__main__":
    main()
