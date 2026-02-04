#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from polideep.plots.uncertainty import plot_error_detection_roc
from polideep.utils.io import ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--out_table", default="tables/uncertainty_error_auc.csv")
    args = ap.parse_args()

    pred = Path(args.pred_dir)
    y = np.load(pred/"y.npy")
    mu = np.load(pred/f"p_mc_mean_T{args.T}.npy")
    yhat = (mu >= 0.5).astype(int)

    out = ensure_dir(Path(args.out_dir))
    rows = []
    for name, arrfile in [("entropy", pred/f"ent_mc_T{args.T}.npy"), ("variance", pred/f"var_mc_T{args.T}.npy")]:
        u = np.load(arrfile)
        auc = plot_error_detection_roc(y, yhat, u, out/f"{args.prefix}uncertainty_error_roc_{name}.png", title=name)
        rows.append({"uncertainty": f"{name}_T{args.T}", "error_auroc": auc})
    df = pd.DataFrame(rows)
    ensure_dir(Path(args.out_table).parent)
    df.to_csv(args.out_table, index=False)
    print(df)

if __name__ == "__main__":
    main()
