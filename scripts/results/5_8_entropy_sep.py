#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from polideep.plots.uncertainty import plot_entropy_sep
from polideep.utils.io import ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()

    pred = Path(args.pred_dir)
    y = np.load(pred/"y.npy")
    mu = np.load(pred/f"p_mc_mean_T{args.T}.npy")
    ent = np.load(pred/f"ent_mc_T{args.T}.npy")
    yhat = (mu >= 0.5).astype(int)

    out = ensure_dir(Path(args.out_dir))
    plot_entropy_sep(y, yhat, ent, out/f"{args.prefix}entropy_correct_vs_error.png", title=f"MC T={args.T}")

if __name__ == "__main__":
    main()
