from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict_proba_positive(model, X) -> np.ndarray:
    """Return probability of class 1 for estimators with predict_proba."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    raise ValueError("Model must support predict_proba or decision_function.")


def classify_with_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    return (proba >= threshold).astype(int)


def evaluate_binary_classifier(y_true, proba, threshold: float = 0.5) -> Dict[str, float]:
    """Compute classification and ranking metrics."""
    y_pred = classify_with_threshold(proba, threshold)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier_score": brier_score_loss(y_true, proba),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def tune_threshold(y_true, proba, thresholds: Iterable[float] | None = None) -> pd.DataFrame:
    """Evaluate thresholds and return a DataFrame sorted by validation F1-score."""
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.01)

    rows = [evaluate_binary_classifier(y_true, proba, float(t)) for t in thresholds]
    return pd.DataFrame(rows).sort_values(by="f1", ascending=False).reset_index(drop=True)
