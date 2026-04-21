"""Generate and save project figures from the heart-disease dataset.

Usage:
    python src/generate_plots.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.advanced_modeling_framework import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    calibration_points,
    export_benchmark_outputs,
    run_stratified_cv_benchmark,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
ADVANCED = REPORTS / "advanced"


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_data() -> pd.DataFrame:
    path = PROCESSED / "heart_cleaned.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `python src/preprocess.py` first.")
    return pd.read_csv(path)


def plot_missing_values(df: pd.DataFrame, out: Path) -> Path:
    counts = df.isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(counts.index, counts.values)
    ax.set_title("Missing values by feature")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, out)
    return out


def plot_target_distribution(df: pd.DataFrame, out: Path) -> Path:
    counts = df["target"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["no_disease", "disease"], counts.values)
    ax.set_title("Target distribution")
    ax.set_ylabel("Count")
    _save(fig, out)
    return out


def plot_numeric_boxplots_by_target(df: pd.DataFrame, out: Path) -> Path:
    n_cols = 3
    n_rows = int(np.ceil(len(NUMERIC_FEATURES) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.6 * n_rows))
    axes = np.array(axes).reshape(-1)

    for idx, col in enumerate(NUMERIC_FEATURES):
        ax = axes[idx]
        grouped = [df.loc[df["target"] == 0, col].dropna(), df.loc[df["target"] == 1, col].dropna()]
        ax.boxplot(grouped, labels=["0", "1"])
        ax.set_title(col)
        ax.set_xlabel("target")

    for ax in axes[len(NUMERIC_FEATURES):]:
        ax.axis("off")

    fig.suptitle("Numeric feature distributions by target", y=1.02)
    _save(fig, out)
    return out


def plot_categorical_distributions_by_target(df: pd.DataFrame, out: Path) -> Path:
    n_cols = 3
    n_rows = int(np.ceil(len(CATEGORICAL_FEATURES) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.8 * n_rows))
    axes = np.array(axes).reshape(-1)

    for idx, col in enumerate(CATEGORICAL_FEATURES):
        ax = axes[idx]
        table = pd.crosstab(df[col], df["target"], normalize="index").fillna(0)
        x = np.arange(len(table.index))
        ax.bar(x, table.get(0, pd.Series(0, index=table.index)), label="target=0")
        ax.bar(x, table.get(1, pd.Series(0, index=table.index)), bottom=table.get(0, 0), label="target=1")
        ax.set_title(col)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in table.index], rotation=45)
        ax.set_ylim(0, 1)

    for ax in axes[len(CATEGORICAL_FEATURES):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Categorical distributions by target (stacked proportions)", y=1.02)
    _save(fig, out)
    return out


def plot_numeric_correlation_heatmap(df: pd.DataFrame, out: Path) -> Path:
    corr = df[NUMERIC_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(NUMERIC_FEATURES)))
    ax.set_yticks(np.arange(len(NUMERIC_FEATURES)))
    ax.set_xticklabels(NUMERIC_FEATURES, rotation=45, ha="right")
    ax.set_yticklabels(NUMERIC_FEATURES)
    ax.set_title("Numeric feature correlation heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, out)
    return out


def plot_roc_pr_curves(oof_predictions: dict[str, pd.DataFrame], out_roc: Path, out_pr: Path) -> tuple[Path, Path]:
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))

    for model_name, frame in oof_predictions.items():
        y_true = frame["y_true"].to_numpy()
        y_prob = frame["y_prob"].to_numpy()

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name=model_name).plot(ax=ax_roc)

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        PrecisionRecallDisplay(
            precision=precision, recall=recall, estimator_name=model_name
        ).plot(ax=ax_pr)

    ax_roc.set_title("ROC curves (OOF)")
    ax_pr.set_title("Precision-Recall curves (OOF)")
    _save(fig_roc, out_roc)
    _save(fig_pr, out_pr)
    return out_roc, out_pr


def plot_cv_metric_bars(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_paths: list[Path] = []
    metrics = ["roc_auc", "pr_auc", "f1", "balanced_accuracy", "brier_score"]

    for metric in metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col not in summary.columns:
            continue

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(summary["model"], summary[mean_col], yerr=summary.get(std_col), capsize=4)
        ax.set_title(f"CV {metric} (mean ± std)")
        ax.tick_params(axis="x", rotation=30)
        out_path = out_dir / f"cv_{metric}_bar.png"
        _save(fig, out_path)
        out_paths.append(out_path)

    return out_paths


def plot_confusion_matrices_top2(summary: pd.DataFrame, oof_predictions: dict[str, pd.DataFrame], out: Path) -> Path:
    top_models = summary["model"].head(2).tolist()
    fig, axes = plt.subplots(1, len(top_models), figsize=(5 * len(top_models), 4))
    axes = np.array(axes).reshape(-1)

    for ax, model_name in zip(axes, top_models):
        frame = oof_predictions[model_name]
        cm = confusion_matrix(frame["y_true"], frame["y_pred_default"], labels=[0, 1])
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1]).plot(ax=ax, colorbar=False)
        ax.set_title(f"{model_name} (threshold=0.5)")

    _save(fig, out)
    return out


def plot_calibration_curves_top3(summary: pd.DataFrame, oof_predictions: dict[str, pd.DataFrame], out: Path) -> Path:
    top_models = summary["model"].head(3).tolist()
    fig, ax = plt.subplots(figsize=(6, 5))

    for model_name in top_models:
        frame = oof_predictions[model_name]
        points = calibration_points(frame["y_true"].to_numpy(), frame["y_prob"].to_numpy(), n_bins=10)
        CalibrationDisplay(
            prob_true=points["frac_pos"].to_numpy(),
            prob_pred=points["mean_pred"].to_numpy(),
            estimator_name=model_name,
        ).plot(ax=ax)

    ax.set_title("Calibration curves (top models, OOF)")
    _save(fig, out)
    return out


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    ADVANCED.mkdir(parents=True, exist_ok=True)

    df = load_data()
    artifacts = run_stratified_cv_benchmark(df)
    export_benchmark_outputs(artifacts, ADVANCED)

    saved: list[Path] = []
    saved.append(plot_missing_values(df, FIGURES / "eda_missing_values.png"))
    saved.append(plot_target_distribution(df, FIGURES / "eda_target_distribution.png"))
    saved.append(plot_numeric_boxplots_by_target(df, FIGURES / "eda_numeric_boxplots_by_target.png"))
    saved.append(
        plot_categorical_distributions_by_target(
            df, FIGURES / "eda_categorical_distributions_by_target.png"
        )
    )
    saved.append(plot_numeric_correlation_heatmap(df, FIGURES / "eda_numeric_correlation_heatmap.png"))

    roc_path, pr_path = plot_roc_pr_curves(
        artifacts.oof_predictions,
        FIGURES / "model_roc_curves.png",
        FIGURES / "model_pr_curves.png",
    )
    saved.extend([roc_path, pr_path])

    saved.extend(plot_cv_metric_bars(artifacts.summary_metrics, FIGURES))
    saved.append(
        plot_confusion_matrices_top2(
            artifacts.summary_metrics,
            artifacts.oof_predictions,
            FIGURES / "model_confusion_matrices_top2.png",
        )
    )
    saved.append(
        plot_calibration_curves_top3(
            artifacts.summary_metrics,
            artifacts.oof_predictions,
            FIGURES / "model_calibration_curves_top3.png",
        )
    )

    pd.DataFrame({"figure_path": [str(p.relative_to(ROOT)) for p in saved]}).to_csv(
        ADVANCED / "figure_manifest.csv", index=False
    )

    print("Saved figures:")
    for p in saved:
        print(f"- {p}")

    if artifacts.unavailable_models:
        print("Unavailable models in this environment:", ", ".join(artifacts.unavailable_models))


if __name__ == "__main__":
    main()
