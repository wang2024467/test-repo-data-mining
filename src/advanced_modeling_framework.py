"""Framework for advanced model comparison and evaluation.

This module is additive: it does not modify the existing training pipeline.
It provides a structured starting point for:
- 5-fold stratified cross-validation benchmarking
- richer metric reporting
- threshold tuning for top models
- calibration analysis helpers
- plot plan placeholders (EDA / model comparison / interpretability / UMAP)
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]


@dataclass
class BenchmarkArtifacts:
    """Outputs from cross-validated benchmarking."""

    fold_metrics: pd.DataFrame
    summary_metrics: pd.DataFrame
    oof_predictions: dict[str, pd.DataFrame]
    unavailable_models: list[str]


def make_preprocessor() -> ColumnTransformer:
    """Build preprocessing consistent with existing project conventions."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_model_registry(random_state: int = 42) -> tuple[dict[str, Any], list[str]]:
    """Return model instances and a list of models unavailable in the env."""
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=400, random_state=random_state, class_weight="balanced"
        ),
        "knn": KNeighborsClassifier(n_neighbors=15),
    }
    unavailable: list[str] = []

    if find_spec("xgboost") is not None:
        xgb_module = import_module("xgboost")
        models["xgboost"] = xgb_module.XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
        )
    else:
        unavailable.append("xgboost")

    if find_spec("catboost") is not None:
        cat_module = import_module("catboost")
        models["catboost"] = cat_module.CatBoostClassifier(
            depth=6,
            learning_rate=0.05,
            iterations=400,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            random_seed=random_state,
        )
    else:
        unavailable.append("catboost")

    return models, unavailable


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute ranking, classification, and calibration metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def run_stratified_cv_benchmark(
    df: pd.DataFrame,
    target_col: str = "target",
    n_splits: int = 5,
    random_state: int = 42,
) -> BenchmarkArtifacts:
    """Evaluate all models with stratified 5-fold CV and return mean±std summaries."""
    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[target_col].astype(int).to_numpy()

    models, unavailable = build_model_registry(random_state=random_state)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_rows: list[dict[str, Any]] = []
    oof_predictions: dict[str, pd.DataFrame] = {}

    for model_name, model in models.items():
        oof_prob = np.zeros_like(y, dtype=float)
        for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
            preprocessor = make_preprocessor()
            pipe = Pipeline([("prep", preprocessor), ("model", model)])
            pipe.fit(x.iloc[train_idx], y[train_idx])

            prob = pipe.predict_proba(x.iloc[valid_idx])[:, 1]
            oof_prob[valid_idx] = prob

            metric_row = {"model": model_name, "fold": fold_idx}
            metric_row.update(compute_metrics(y[valid_idx], prob, threshold=0.5))
            fold_rows.append(metric_row)

        oof_predictions[model_name] = pd.DataFrame(
            {"y_true": y, "y_prob": oof_prob, "y_pred_default": (oof_prob >= 0.5).astype(int)}
        )

    fold_metrics = pd.DataFrame(fold_rows)
    summary = (
        fold_metrics.groupby("model")
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    summary = summary.sort_values(by="roc_auc_mean", ascending=False).reset_index(drop=True)

    return BenchmarkArtifacts(
        fold_metrics=fold_metrics,
        summary_metrics=summary,
        oof_predictions=oof_predictions,
        unavailable_models=unavailable,
    )


def threshold_search(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    grid: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    """Search threshold grid and return best threshold + full threshold table."""
    if grid is None:
        grid = np.arange(0.10, 0.91, 0.01)

    rows: list[dict[str, float]] = []
    for threshold in grid:
        row = {"threshold": float(threshold)}
        row.update(compute_metrics(y_true, y_prob, threshold=float(threshold)))
        rows.append(row)

    table = pd.DataFrame(rows)
    if metric not in table.columns:
        raise ValueError(f"Unsupported metric for threshold search: {metric}")

    best_idx = table[metric].idxmax()
    return float(table.loc[best_idx, "threshold"]), table


def calibration_points(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return points for calibration curve plotting."""
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_pred": mean_pred, "frac_pos": frac_pos})


def export_benchmark_outputs(artifacts: BenchmarkArtifacts, out_dir: Path) -> None:
    """Persist CV summary, fold metrics, and OOF predictions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts.fold_metrics.to_csv(out_dir / "cv_fold_metrics.csv", index=False)
    artifacts.summary_metrics.to_csv(out_dir / "cv_summary_metrics.csv", index=False)

    for model_name, frame in artifacts.oof_predictions.items():
        frame.to_csv(out_dir / f"oof_{model_name}.csv", index=False)

    if artifacts.unavailable_models:
        pd.DataFrame({"unavailable_model": artifacts.unavailable_models}).to_csv(
            out_dir / "unavailable_models.csv", index=False
        )


PLOT_PLAN = {
    "eda": [
        "missing_value_barplot",
        "target_distribution",
        "numeric_by_target_boxplots",
        "categorical_by_target_bars",
        "numeric_correlation_heatmap",
    ],
    "model_comparison": [
        "roc_curves_all_models",
        "pr_curves_all_models",
        "cv_metric_barplots",
        "confusion_matrices_top2",
        "calibration_curves_top_models",
    ],
    "interpretability": [
        "logreg_coefficient_plot",
        "tree_or_boosting_feature_importance",
        "optional_shap_for_best_boosting_model",
    ],
    "umap": [
        "umap_true_label",
        "umap_predicted_probability",
        "umap_error_type_tp_tn_fp_fn",
    ],
}
