# CS439 Final Project: Explainable Credit Card Default Risk Prediction

This repository contains a reproducible machine learning pipeline for predicting next-month credit card default risk using the UCI **Default of Credit Card Clients** dataset.

## Project Title

**Beyond Accuracy: Explainable and Calibrated Credit Card Default Risk Prediction**

## Research Question

Can feature engineering, class-imbalance-aware model training, threshold tuning, and calibration improve both predictive performance and interpretability for credit card default risk prediction?

## Dataset

- Source: UCI Machine Learning Repository, dataset ID 350
- Dataset: Default of Credit Card Clients
- Instances: 30,000 clients
- Features: 23 explanatory variables
- Target: next-month default payment, where 1 means default and 0 means no default

The data is downloaded automatically with `ucimlrepo`, so the raw dataset is not committed to the repository.

## Main Pipeline

1. Download/load dataset from UCI.
2. Clean categorical values for `EDUCATION`, `MARRIAGE`, and `SEX`.
3. Engineer risk-oriented features:
   - average bill amount
   - average payment amount
   - bill utilization ratios
   - payment-to-bill ratios
   - delinquency count
   - maximum and average delay
   - recent bill trend
4. Train baseline models:
   - Logistic Regression
   - Random Forest
   - HistGradientBoosting
5. Tune the classification threshold on a validation set.
6. Evaluate on a held-out test set with:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC
   - PR-AUC
   - Brier score
7. Generate visualizations:
   - ROC curve
   - Precision-recall curve
   - Confusion matrix
   - Calibration curve
   - Permutation feature importance
8. Optional explainability analysis using SHAP.

## Repository Structure

```text
cs439_credit_default_project/
├── data/
│   ├── raw/
│   └── processed/
├── figures/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling_results.ipynb
├── report/
│   ├── main.tex
│   └── references.bib
├── results/
├── src/
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── evaluate.py
│   ├── plots.py
│   ├── explain.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

## Run the Full Experiment

```bash
python -m src.train
```

After running, check:

- `results/metrics.csv`
- `results/thresholds.csv`
- `figures/roc_curve.png`
- `figures/pr_curve.png`
- `figures/confusion_matrix.png`
- `figures/calibration_curve.png`
- `figures/permutation_importance.png`

## Report

The starter report is in:

```text
report/main.tex
```

It is written in a NeurIPS-style academic-paper structure. Replace the placeholder result numbers after you run the code.

## Reproducibility Notes

- Random seed is fixed at `42`.
- The test set is held out until final evaluation.
- Scaling and one-hot encoding are fitted only on the training split through an sklearn `Pipeline` and `ColumnTransformer`.
- Threshold tuning is performed on a validation split, not on the test set.
- The raw dataset is downloaded from the official UCI source instead of being manually edited.
