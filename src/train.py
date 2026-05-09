from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import RANDOM_STATE, load_local_or_download, split_X_y
from src.evaluate import evaluate_binary_classifier, predict_proba_positive, tune_threshold
from src.explain import try_make_shap_summary
from src.features import add_engineered_features
from src.models import build_model_pipelines
from src.plots import (
    plot_calibration,
    plot_confusion,
    plot_permutation_importance,
    plot_roc_pr_curves,
)


def main() -> None:
    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    df = load_local_or_download()
    X_raw, y = split_X_y(df)
    X = add_engineered_features(X_raw)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        stratify=y_train_full,
        random_state=RANDOM_STATE,
    )

    X_train_full.to_csv("data/processed/X_train_full.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train_full.to_frame("default_next_month").to_csv("data/processed/y_train_full.csv", index=False)
    y_test.to_frame("default_next_month").to_csv("data/processed/y_test.csv", index=False)

    pipelines = build_model_pipelines(X_train)

    rows = []
    threshold_rows = []
    val_probabilities = {}
    test_probabilities = {}
    fitted_models = {}

    for name, model in pipelines.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model

        val_proba = predict_proba_positive(model, X_val)
        test_proba = predict_proba_positive(model, X_test)
        val_probabilities[name] = val_proba
        test_probabilities[name] = test_proba

        # Standard 0.50 threshold result.
        standard_metrics = evaluate_binary_classifier(y_test, test_proba, threshold=0.50)
        standard_metrics["model"] = name
        standard_metrics["setting"] = "threshold_0.50"
        rows.append(standard_metrics)

        # Validation-only threshold tuning.
        threshold_table = tune_threshold(y_val, val_proba)
        threshold_table.insert(0, "model", name)
        threshold_rows.append(threshold_table)
        best_threshold = float(threshold_table.iloc[0]["threshold"])

        tuned_metrics = evaluate_binary_classifier(y_test, test_proba, threshold=best_threshold)
        tuned_metrics["model"] = name
        tuned_metrics["setting"] = "validation_tuned_threshold"
        rows.append(tuned_metrics)

    metrics = pd.DataFrame(rows)
    metrics = metrics[
        [
            "model",
            "setting",
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "brier_score",
            "tn",
            "fp",
            "fn",
            "tp",
        ]
    ]
    metrics.to_csv("results/metrics.csv", index=False)
    pd.concat(threshold_rows, ignore_index=True).to_csv("results/thresholds.csv", index=False)

    print("\nFinal test metrics:")
    print(metrics.sort_values(["roc_auc", "pr_auc"], ascending=False).to_string(index=False))

    # Select the model with best test PR-AUC among tuned-threshold settings for plots.
    tuned = metrics[metrics["setting"] == "validation_tuned_threshold"].copy()
    best_row = tuned.sort_values("pr_auc", ascending=False).iloc[0]
    best_model_name = best_row["model"]
    best_threshold = float(best_row["threshold"])
    best_model = fitted_models[best_model_name]
    best_test_proba = test_probabilities[best_model_name]

    plot_roc_pr_curves(y_test, test_probabilities)
    plot_confusion(y_test, best_test_proba, best_threshold)
    plot_calibration(y_test, test_probabilities)
    plot_permutation_importance(best_model, X_test, y_test)

    # Optional SHAP plot on a small sample.
    X_sample = X_test.sample(n=min(500, len(X_test)), random_state=RANDOM_STATE)
    try_make_shap_summary(best_model, X_sample)

    print(f"\nBest model for detailed plots: {best_model_name} at threshold {best_threshold:.2f}")
    print("Outputs saved in results/ and figures/.")


if __name__ == "__main__":
    main()
