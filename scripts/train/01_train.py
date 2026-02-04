#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from polideep.data.dataset import ImageCSVDataset
from polideep.models.backbones import BinaryClassifier, ModelConfig
from polideep.train import TrainConfig, train_model
from polideep.utils.seed import set_all_seeds, SeedConfig
from polideep.utils.device import get_default_device

def build_transforms(res: int, imagenet_norm: bool):
    t = [transforms.Resize((res, res)), transforms.ToTensor()]
    if imagenet_norm:
        t.append(transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))
    return transforms.Compose(t)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/political_openfake")
    ap.add_argument("--splits_dir", default="data/political_openfake/splits")
    ap.add_argument("--backbone", choices=["resnet18","efficientnet_b4"], required=True)
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--resolution", type=int, default=380)
    ap.add_argument("--imagenet_norm", action="store_true")
    ap.add_argument("--dropout_p", type=float, default=0.0)
    args = ap.parse_args()

    set_all_seeds(SeedConfig(seed=args.seed))
    device = get_default_device()

    tfm = build_transforms(args.resolution, args.imagenet_norm)

    train_csv = Path(args.splits_dir) / "train.csv"
    val_csv = Path(args.splits_dir) / "val.csv"

    train_ds = ImageCSVDataset(args.data_root, train_csv, transform=tfm)
    val_ds = ImageCSVDataset(args.data_root, val_csv, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = BinaryClassifier(ModelConfig(backbone=args.backbone, pretrained=True, dropout_p=args.dropout_p))
    run_dir = Path(args.out_dir) / f"{args.backbone}_seed{args.seed}_res{args.resolution}" / "train"
    info = train_model(model, train_loader, val_loader, run_dir, TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_workers=args.num_workers,
        device=device,
    ))
    print(info)

if __name__ == "__main__":
    main()
