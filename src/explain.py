from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def try_make_shap_summary(model, X_sample: pd.DataFrame, output_dir="figures") -> bool:
    """Create a SHAP summary plot when shap supports the fitted model.

    This is optional because SHAP support can vary by model and package version.
    The main project already produces permutation importance as a stable fallback.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import shap

        transformed = model.named_steps["preprocess"].transform(X_sample)
        feature_names = model.named_steps["preprocess"].get_feature_names_out()
        estimator = model.named_steps["model"]

        explainer = shap.Explainer(estimator, transformed)
        shap_values = explainer(transformed)

        shap.summary_plot(shap_values, transformed, feature_names=feature_names, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary.png", dpi=200)
        plt.close()
        return True
    except Exception as exc:  # optional analysis should not break full experiment
        print(f"SHAP summary skipped: {exc}")
        return False
