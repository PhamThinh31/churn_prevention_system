"""Churn evaluation metrics — uniform interface across the three model types."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(y_true: np.ndarray, y_score: np.ndarray, threshold: float | None = None) -> dict:
    """Return ranking + calibration + threshold-dependent metrics.

    Ranking:        auc_roc, pr_auc (== average_precision_score, area under the PR curve)
    Calibration:    brier_score (mean squared error of probability vs label, lower is better)
    Threshold-dep:  precision, recall, f1 at best-F1 threshold (or the supplied one)
    """
    auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))
    brier = float(brier_score_loss(y_true, np.clip(y_score, 0.0, 1.0)))

    if threshold is None:
        prec, rec, thr = precision_recall_curve(y_true, y_score)
        f1s = 2 * prec * rec / (prec + rec + 1e-12)
        best = int(np.nanargmax(f1s[:-1])) if len(thr) else 0
        threshold = float(thr[best]) if len(thr) else 0.5

    y_pred = (y_score >= threshold).astype("int8")
    return {
        "auc_roc": auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(y_pred.mean()),
        "base_rate": float(y_true.mean()),
    }


def compare(rows: list[dict]) -> pd.DataFrame:
    """Pretty comparison table across multiple (name, metrics dict) rows."""
    return pd.DataFrame(rows).set_index("name").round(4)
