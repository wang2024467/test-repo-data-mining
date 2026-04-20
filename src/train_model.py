"""Train and evaluate heart disease prediction models.

Usage:
    python src/train_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def load_data() -> pd.DataFrame:
    path = PROCESSED / "heart_cleaned.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python src/preprocess.py` first."
        )
    return pd.read_csv(path)


def get_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]
    return df[features].copy(), df["target"].astype(int)


def build_preprocessor() -> ColumnTransformer:
    numeric = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    categorical = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def evaluate_pipeline(name: str, pipe: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    probas = pipe.predict_proba(x_test)[:, 1]
    preds = (probas >= 0.5).astype(int)
    result = {
        "model": name,
        "roc_auc": float(roc_auc_score(y_test, probas)),
        "pr_auc": float(average_precision_score(y_test, probas)),
        "classification_report": classification_report(y_test, preds, output_dict=True),
    }
    return result


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    df = load_data()
    x, y = get_features_target(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=400, random_state=42, class_weight="balanced"
        ),
    }

    results = []
    for name, model in models.items():
        pipeline = Pipeline([("prep", preprocessor), ("model", model)])
        pipeline.fit(x_train, y_train)
        results.append(evaluate_pipeline(name, pipeline, x_test, y_test))

    results_path = REPORTS / "metrics.json"
    results_path.write_text(json.dumps(results, indent=2))

    leaderboard = sorted(results, key=lambda r: r["roc_auc"], reverse=True)
    best = leaderboard[0]

    print("Saved metrics:", results_path)
    print("Best model:", best["model"])
    print("ROC-AUC:", round(best["roc_auc"], 4))
    print("PR-AUC:", round(best["pr_auc"], 4))


if __name__ == "__main__":
    main()
