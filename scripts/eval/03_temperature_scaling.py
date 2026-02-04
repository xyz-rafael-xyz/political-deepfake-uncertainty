#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from polideep.calibration.temperature import fit_temperature, apply_temperature
from polideep.utils.io import write_json

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_pred_dir", required=True)
    ap.add_argument("--test_pred_dir", required=True)
    ap.add_argument("--out_name", default="p_temp.npy")
    args = ap.parse_args()

    val = Path(args.val_pred_dir)
    test = Path(args.test_pred_dir)

    yv = np.load(val/"y.npy")
    lv = np.load(val/"logits_det.npy")
    tau = fit_temperature(lv, yv, device="cpu")
    lt = np.load(test/"logits_det.npy")
    lt_cal = apply_temperature(lt, tau)
    pt = sigmoid(lt_cal)
    np.save(test/args.out_name, pt)
    write_json({"tau": tau}, test/"temp_scaling.json")
    print(f"tau={tau:.6f} saved {args.out_name} in {test}")

if __name__ == "__main__":
    main()
