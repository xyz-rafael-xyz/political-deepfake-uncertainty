from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

@dataclass(frozen=True)
class EvalSemantics:
    positive_label: int = 1
    threshold: float = 0.5

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())

def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(y_true, scores))

def confusion(y_true: np.ndarray, scores: np.ndarray, thr: float = 0.5) -> Dict[str, int]:
    y_pred = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}

def roc_points(y_true: np.ndarray, scores: np.ndarray):
    fpr, tpr, thr = roc_curve(y_true, scores)
    return fpr, tpr, thr
