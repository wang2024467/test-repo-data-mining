# Heart Disease ML + AI Portfolio Project

This repository is now structured as an end-to-end project (not only data mining):
1. **Data preprocessing/cleaning** (Python + SQL)
2. **Machine learning model training/evaluation**
3. **LLM-ready insight generation from model outputs**

## Recommended GitHub Structure

```text
heart-disease-project/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                  # Original files (.data, .names, etc.)
│   ├── interim/              # Temporary merged datasets
│   └── processed/            # Final cleaned datasets
├── reports/
│   ├── metrics.json          # Model performance output
│   └── llm_brief.md          # Prompt/brief for LLM-based interpretation
├── notebooks/
│   └── 01_eda.ipynb          # Optional EDA notebook
├── sql/
│   └── heart_cleaning.sql    # SQL cleaning workflow (DuckDB)
├── src/
│   ├── preprocess.py         # Data preprocessing pipeline
│   ├── train_model.py        # ML training and evaluation
│   └── generate_llm_brief.py # Build LLM-ready summary from metrics
└── tests/
    └── test_preprocess.py
```

## Which files to use first

Start with:
- `processed.cleveland.data`
- `processed.hungarian.data`
- `processed.switzerland.data`
- `processed.va.data`

Put those files into `data/raw/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 1: Preprocess data (Python)

```bash
python src/preprocess.py
```

Outputs:
- `data/interim/heart_combined_raw.csv`
- `data/processed/heart_cleaned.csv`
- `data/processed/heart_model_ready.csv`

## Step 1B (Optional): Show SQL skills

Use DuckDB to run SQL preprocessing:

```bash
duckdb -c ".read sql/heart_cleaning.sql"
```

Output:
- `data/processed/heart_cleaned_sql.csv`

## Step 2: Train ML models

```bash
python src/train_model.py
```

What it does:
- train/test split with stratification
- preprocess with imputation + one-hot encoding
- compares Logistic Regression and Random Forest
- reports ROC-AUC and PR-AUC

Output:
- `reports/metrics.json`

## Step 3: Use LLM for decision-support narrative

```bash
python src/generate_llm_brief.py
```

Output:
- `reports/llm_brief.md`

You can paste `reports/llm_brief.md` into ChatGPT or any enterprise LLM to generate:
- model comparison narrative,
- deployment recommendation,
- risk/ethics caveats,
- next experiments for improvement.

## Why this is strong for your job profile

You demonstrate:
- **Python data engineering** (cleaning and feature prep)
- **SQL transformation skills**
- **ML modeling and evaluation**
- **AI/LLM integration for interpretability and communication**

