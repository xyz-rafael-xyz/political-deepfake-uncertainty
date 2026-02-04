from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

from ..metrics.classification import roc_points
from ..metrics.calibration import ece, ECEConfig

def savefig(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_roc(y: np.ndarray, scores: np.ndarray, out_path: str | Path, title: str = "") -> None:
    fpr, tpr, _ = roc_points(y, scores)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if title:
        plt.title(title)
    savefig(out_path)

def plot_score_hist(y: np.ndarray, scores: np.ndarray, out_path: str | Path, title: str = "") -> None:
    plt.figure()
    plt.hist(scores[y==0], bins=40, alpha=0.6, label="real (y=0)")
    plt.hist(scores[y==1], bins=40, alpha=0.6, label="fake (y=1)")
    plt.axvline(0.5, linestyle="--")
    plt.xlabel("p(y=1|x)")
    plt.ylabel("count")
    plt.legend()
    if title:
        plt.title(title)
    savefig(out_path)

def plot_reliability(y: np.ndarray, p: np.ndarray, out_path: str | Path, title: str = "", n_bins: int = 15) -> Dict[str, np.ndarray]:
    e, b = ece(y, p, ECEConfig(n_bins=n_bins))
    edges = b["bin_edges"]
    acc = b["bin_acc"]
    conf = b["bin_conf"]
    cnt = b["bin_count"]

    plt.figure()
    plt.plot([0,1],[0,1], linestyle="--")
    plt.plot(conf, acc, marker="o")
    plt.xlabel("confidence")
    plt.ylabel("accuracy")
    if title:
        plt.title(f"{title} (ECE={e:.4f})")
    savefig(out_path)
    return b
