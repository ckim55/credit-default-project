from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from ucimlrepo import fetch_ucirepo

RANDOM_STATE = 42
TARGET_COL = "default_next_month"

COLUMN_MAP = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_0",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    "X12": "BILL_AMT1",
    "X13": "BILL_AMT2",
    "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",
    "X16": "BILL_AMT5",
    "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",
    "X19": "PAY_AMT2",
    "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",
    "X22": "PAY_AMT5",
    "X23": "PAY_AMT6",
    "Y": TARGET_COL,
    "default payment next month": TARGET_COL,
    "default.payment.next.month": TARGET_COL,
}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename UCI variable names into readable project names."""
    renamed = df.rename(columns={c: COLUMN_MAP.get(c, c) for c in df.columns})
    renamed.columns = [str(c).strip() for c in renamed.columns]
    return renamed


def load_uci_credit_default() -> pd.DataFrame:
    """Download the UCI credit default dataset and return one clean DataFrame."""
    dataset = fetch_ucirepo(id=350)
    X = normalize_column_names(dataset.data.features.copy())
    y = normalize_column_names(dataset.data.targets.copy())

    if y.shape[1] != 1:
        raise ValueError(f"Expected exactly one target column, got {list(y.columns)}")

    y = y.rename(columns={y.columns[0]: TARGET_COL})
    df = pd.concat([X, y], axis=1)
    return normalize_column_names(df)


def save_raw_dataset(path: str | Path = "data/raw/credit_default_raw.csv") -> pd.DataFrame:
    """Load the dataset and save a CSV copy locally for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = load_uci_credit_default()
    df.to_csv(path, index=False)
    return df


def load_local_or_download(path: str | Path = "data/raw/credit_default_raw.csv") -> pd.DataFrame:
    """Use a cached local CSV if available; otherwise download from UCI."""
    path = Path(path)
    if path.exists():
        return normalize_column_names(pd.read_csv(path))
    return save_raw_dataset(path)


def split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features and target."""
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    return X, y
