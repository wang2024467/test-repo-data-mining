"""Preprocess UCI heart-disease processed datasets.

Usage:
    python src/preprocess.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


COLUMNS = [
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
    "num",
]

PROCESSED_FILES = [
    "processed.cleveland.data",
    "processed.hungarian.data",
    "processed.switzerland.data",
    "processed.va.data",
]


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"



def read_source_files() -> pd.DataFrame:
    """Read and concatenate all available processed data files."""
    frames: list[pd.DataFrame] = []

    for file_name in PROCESSED_FILES:
        path = RAW_DIR / file_name
        if not path.exists():
            continue

        df = pd.read_csv(path, header=None, names=COLUMNS)
        df["source_file"] = file_name
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No processed source files found in data/raw/. "
            "Expected one or more of: "
            f"{', '.join(PROCESSED_FILES)}"
        )

    return pd.concat(frames, ignore_index=True)



def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning and type harmonization."""
    cleaned = df.copy()

    cleaned = cleaned.replace("?", pd.NA)

    numeric_cols = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
        "ca",
        "num",
    ]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned["target"] = (cleaned["num"] > 0).astype("Int64")

    return cleaned



def encode_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values and one-hot encode categorical features."""
    feature_cols = [
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

    x = df[feature_cols].copy()
    # scikit-learn imputers expect np.nan rather than pandas.NA in object arrays.
    x = x.replace({pd.NA: np.nan})

    y = df["target"].copy()

    numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    categorical_features = [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "thal",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[("imputer", SimpleImputer(strategy="median"))]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    x_t = preprocessor.fit_transform(x)

    cat_columns = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(
        categorical_features
    )
    final_columns = numeric_features + list(cat_columns)

    out = pd.DataFrame(x_t, columns=final_columns)
    out["target"] = y.to_numpy()

    return out



def main() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = read_source_files()
    raw_df.to_csv(INTERIM_DIR / "heart_combined_raw.csv", index=False)

    cleaned_df = clean_dataframe(raw_df)
    cleaned_df.to_csv(PROCESSED_DIR / "heart_cleaned.csv", index=False)

    model_df = encode_for_model(cleaned_df)
    model_df.to_csv(PROCESSED_DIR / "heart_model_ready.csv", index=False)

    print("Saved:")
    print(f"- {INTERIM_DIR / 'heart_combined_raw.csv'}")
    print(f"- {PROCESSED_DIR / 'heart_cleaned.csv'}")
    print(f"- {PROCESSED_DIR / 'heart_model_ready.csv'}")


if __name__ == "__main__":
    main()
