#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from polideep.eval import evaluate_binary, EvalConfig
from polideep.utils.io import ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="directory containing y.npy and p_*.npy")
    ap.add_argument("--out_csv", default="tables/metrics_table.csv")
    ap.add_argument("--ece_bins", type=int, default=15)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    pred = Path(args.pred_dir)
    y = np.load(pred/"y.npy")
    out_rows = []
    cfg = EvalConfig(threshold=args.threshold, ece_bins=args.ece_bins)

    for name in sorted(pred.glob("p_*.npy")):
        key = name.stem.replace("p_", "")
        p = np.load(name)
        m = evaluate_binary(y, p, cfg)
        out_rows.append({"Method": key, **m})

    df = pd.DataFrame(out_rows)
    ensure_dir(Path(args.out_csv).parent)
    df.to_csv(args.out_csv, index=False)
    print(df)

if __name__ == "__main__":
    main()
