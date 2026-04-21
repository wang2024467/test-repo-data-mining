# Advanced Modeling Scope (Additive)

This extension is intentionally additive and keeps the existing workflow unchanged:

- Existing scripts remain intact:
  - `src/preprocess.py`
  - `src/train_model.py`
  - `src/generate_model_brief.py`
- New framework module:
  - `src/advanced_modeling_framework.py`

## What is included now

1. Model registry for:
   - Logistic Regression
   - Random Forest
   - KNN
   - XGBoost (if installed)
   - CatBoost (if installed)
2. Stratified 5-fold cross-validation benchmarking.
3. Mean/std metric summaries for:
   - ROC-AUC
   - PR-AUC
   - F1
   - Balanced Accuracy
   - Precision
   - Recall
   - Specificity
   - Brier Score
4. Threshold search utility for best-model postprocessing.
5. Calibration points helper.
6. Plot plan checklist for EDA/model/interpretability/UMAP outputs.

## Suggested next step

Create a dedicated runner script (for example `src/run_advanced_benchmark.py`) that:
- loads `data/processed/heart_cleaned.csv`
- calls `run_stratified_cv_benchmark`
- exports outputs to `reports/advanced/`
- generates requested plots from the plan.
