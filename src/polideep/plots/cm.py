from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from .figures import savefig

def plot_confusion(y: np.ndarray, p: np.ndarray, out_path: str | Path, thr: float = 0.5, title: str = "") -> None:
    yhat = (p >= thr).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0,1])
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.xticks([0,1], ["real(0)", "fake(1)"])
    plt.yticks([0,1], ["real(0)", "fake(1)"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i,j]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if title:
        plt.title(title)
    savefig(out_path)
