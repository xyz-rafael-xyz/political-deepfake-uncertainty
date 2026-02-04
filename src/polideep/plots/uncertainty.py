from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from .figures import savefig

def plot_entropy_sep(y: np.ndarray, yhat: np.ndarray, ent: np.ndarray, out_path: str | Path, title: str = "") -> None:
    correct = (yhat == y)
    plt.figure()
    plt.hist(ent[correct], bins=40, alpha=0.6, label="correct")
    plt.hist(ent[~correct], bins=40, alpha=0.6, label="error")
    plt.xlabel("predictive entropy")
    plt.ylabel("count")
    plt.legend()
    if title:
        plt.title(title)
    savefig(out_path)

def plot_error_detection_roc(y: np.ndarray, yhat: np.ndarray, u: np.ndarray, out_path: str | Path, title: str = "") -> float:
    err = (yhat != y).astype(int)
    auc = roc_auc_score(err, u)
    fpr, tpr, _ = roc_curve(err, u)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if title:
        plt.title(f"{title} (AUROC={auc:.4f})")
    savefig(out_path)
    return float(auc)
