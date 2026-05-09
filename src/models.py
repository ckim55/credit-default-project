from __future__ import annotations

from typing import Dict

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor

RANDOM_STATE = 42


def build_model_pipelines(X_train) -> Dict[str, Pipeline]:
    """Construct baseline and stronger model pipelines."""
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=None,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        ),
    }

    pipelines = {}
    for name, model in models.items():
        # Each pipeline receives its own preprocessor instance so fitted state is not shared.
        pipelines[name] = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X_train)),
                ("model", model),
            ]
        )
    return pipelines
