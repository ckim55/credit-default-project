from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_BASE = [
    "LIMIT_BAL",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]

CATEGORICAL_BASE = ["SEX", "EDUCATION", "MARRIAGE"]
PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
PAYMENT_COLS = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]


def clean_categorical_values(X: pd.DataFrame) -> pd.DataFrame:
    """Clean unusual category codes while preserving the original meaning."""
    X = X.copy()

    if "EDUCATION" in X.columns:
        # UCI codebook lists 1,2,3,4; real data may include 0,5,6.
        # We group 0,5,6 into 4 = others/unknown.
        X["EDUCATION"] = X["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    if "MARRIAGE" in X.columns:
        # Codebook lists 1,2,3; real data may include 0.
        X["MARRIAGE"] = X["MARRIAGE"].replace({0: 3})

    for col in CATEGORICAL_BASE:
        if col in X.columns:
            X[col] = X[col].astype("category")

    return X


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create row-level financial risk features without using the target label."""
    X = clean_categorical_values(X)
    X = X.copy()
    eps = 1e-6

    available_bill_cols = [c for c in BILL_COLS if c in X.columns]
    available_payment_cols = [c for c in PAYMENT_COLS if c in X.columns]
    available_pay_status_cols = [c for c in PAY_STATUS_COLS if c in X.columns]

    if available_bill_cols:
        X["BILL_MEAN"] = X[available_bill_cols].mean(axis=1)
        X["BILL_STD"] = X[available_bill_cols].std(axis=1)
        X["BILL_MAX"] = X[available_bill_cols].max(axis=1)
        X["BILL_TREND_RECENT_MINUS_OLD"] = X["BILL_AMT1"] - X["BILL_AMT6"]

        if "LIMIT_BAL" in X.columns:
            X["UTILIZATION_MEAN"] = X["BILL_MEAN"] / (X["LIMIT_BAL"].abs() + eps)
            X["UTILIZATION_MAX"] = X["BILL_MAX"] / (X["LIMIT_BAL"].abs() + eps)

    if available_payment_cols:
        X["PAYMENT_MEAN"] = X[available_payment_cols].mean(axis=1)
        X["PAYMENT_STD"] = X[available_payment_cols].std(axis=1)
        X["PAYMENT_MAX"] = X[available_payment_cols].max(axis=1)

    if available_bill_cols and available_payment_cols:
        for i, (bill_col, pay_col) in enumerate(zip(BILL_COLS, PAYMENT_COLS), start=1):
            if bill_col in X.columns and pay_col in X.columns:
                X[f"PAY_TO_BILL_RATIO_{i}"] = X[pay_col] / (X[bill_col].abs() + eps)
        ratio_cols = [c for c in X.columns if c.startswith("PAY_TO_BILL_RATIO_")]
        X["PAY_TO_BILL_RATIO_MEAN"] = X[ratio_cols].mean(axis=1)

    if available_pay_status_cols:
        # In this dataset, positive payment status indicates delayed payment.
        X["DELINQUENCY_COUNT"] = (X[available_pay_status_cols] > 0).sum(axis=1)
        X["MAX_DELAY"] = X[available_pay_status_cols].max(axis=1)
        X["AVG_DELAY"] = X[available_pay_status_cols].mean(axis=1)
        X["RECENT_DELAY"] = X["PAY_0"]

    return X


def get_feature_groups(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return numeric and categorical feature lists after feature engineering."""
    categorical_cols = [c for c in CATEGORICAL_BASE if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols and c != "ID"]
    return numeric_cols, categorical_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a preprocessing transformer fitted only inside sklearn pipelines."""
    numeric_cols, categorical_cols = get_feature_groups(X)

    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", one_hot, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor
