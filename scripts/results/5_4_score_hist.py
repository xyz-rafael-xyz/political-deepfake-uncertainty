#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from polideep.plots.figures import plot_score_hist
from polideep.utils.io import ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--method", default="det")
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()

    pred = Path(args.pred_dir)
    y = np.load(pred/"y.npy")
    p = np.load(pred/f"p_{args.method}.npy") if (pred/f"p_{args.method}.npy").exists() else np.load(pred/"p_det.npy")
    out = ensure_dir(Path(args.out_dir))
    plot_score_hist(y, p, out / f"{args.prefix}hist_scores_{args.method}.png", title=args.method)

if __name__ == "__main__":
    main()
