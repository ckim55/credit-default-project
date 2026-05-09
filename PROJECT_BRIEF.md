# Project Brief

## Title
Beyond Accuracy: Explainable and Calibrated Credit Card Default Risk Prediction

## Main Idea
This project predicts whether a credit card customer will default next month. Instead of only training one classifier, the project builds a full data science pipeline with preprocessing, feature engineering, baseline comparison, validation-based threshold tuning, calibration analysis, and model explainability.

## Why This Is Strong for CS439
The assignment asks for a research-driven data science problem, careful preprocessing, technical methodology, experiments, visualizations, and a reproducible GitHub repository. This project naturally covers all of those parts.

## Research Question
Can feature engineering, class-imbalance-aware training, threshold tuning, and explainability improve both predictive performance and interpretability in credit card default risk prediction?

## Hypothesis
Tree-based models with engineered behavioral risk features will outperform a simple logistic regression baseline in ROC-AUC and PR-AUC. Threshold tuning should improve F1-score or recall compared to the default 0.50 threshold.

## Dataset
UCI Default of Credit Card Clients dataset.

- 30,000 clients
- 23 features
- Binary target: default next month
- No missing values listed by UCI
- Includes credit limit, demographic variables, repayment status, bill amounts, and payment amounts

## Models
1. Logistic Regression
2. Random Forest
3. HistGradientBoosting

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Brier score
- Confusion matrix
- Calibration curve

## Visualizations
- ROC curve
- Precision-recall curve
- Confusion matrix
- Calibration curve
- Permutation feature importance
- Optional SHAP summary plot

## Main Claim for the Final Paper
A strong model for default prediction should not be judged only by accuracy. For financial risk prediction, threshold choice, recall, precision, calibration, and interpretability are also important because different types of errors have different practical costs.

## Project Scope and Relevance

This project is designed as a complete data science workflow that includes problem formulation, data preprocessing, feature engineering, model comparison, threshold tuning, evaluation, visualization, and reproducibility. The goal is to evaluate credit default prediction not only through accuracy, but also through recall, precision, calibration, and interpretability.
