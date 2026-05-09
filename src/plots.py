from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

from src.evaluate import classify_with_threshold


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_roc_pr_curves(y_true, model_probabilities: dict, output_dir="figures") -> None:
    output_dir = _ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, proba in model_probabilities.items():
        RocCurveDisplay.from_predictions(y_true, proba, name=name, ax=ax)
    ax.set_title("ROC Curves")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, proba in model_probabilities.items():
        PrecisionRecallDisplay.from_predictions(y_true, proba, name=name, ax=ax)
    ax.set_title("Precision-Recall Curves")
    fig.tight_layout()
    fig.savefig(output_dir / "pr_curve.png", dpi=200)
    plt.close(fig)


def plot_confusion(y_true, proba, threshold: float, output_dir="figures") -> None:
    output_dir = _ensure_dir(output_dir)
    y_pred = classify_with_threshold(proba, threshold)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix at Threshold={threshold:.2f}")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()


def plot_calibration(y_true, model_probabilities: dict, output_dir="figures") -> None:
    output_dir = _ensure_dir(output_dir)

    plt.figure(figsize=(7, 5))
    for name, proba in model_probabilities.items():
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="uniform")
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
    plt.plot([0, 1], [0, 1], linestyle="--", label="perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_curve.png", dpi=200)
    plt.close()


def plot_permutation_importance(model, X_test, y_test, output_dir="figures", top_n: int = 15) -> None:
    output_dir = _ensure_dir(output_dir)
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=8,
        random_state=42,
        n_jobs=-1,
        scoring="average_precision",
    )

    importances = result.importances_mean
    indices = np.argsort(importances)[-top_n:]
    labels = np.array(X_test.columns)[indices]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), labels)
    plt.xlabel("Mean decrease in PR-AUC")
    plt.title("Permutation Feature Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "permutation_importance.png", dpi=200)
    plt.close()
