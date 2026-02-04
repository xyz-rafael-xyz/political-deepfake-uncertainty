#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from polideep.plots.cm import plot_confusion
from polideep.utils.io import ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()

    pred = Path(args.pred_dir)
    y = np.load(pred/"y.npy")
    p = np.load(pred/"p_det.npy")

    out = ensure_dir(Path(args.out_dir))
    plot_confusion(y, p, out / f"{args.prefix}confusion_matrix_det.png", thr=args.thr, title="Deterministic")

if __name__ == "__main__":
    main()
