# Heart Disease ML + AI Portfolio Project

This repository is now structured as an end-to-end project (not only data mining):
1. **Data preprocessing/cleaning** (Python + SQL)
2. **Machine learning model training/evaluation**
3. **Model insight generation from model outputs**

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
│   └── model_brief.md        # Prompt/brief for model interpretation
├── notebooks/
│   └── 01_eda.ipynb          # Optional EDA notebook
├── sql/
│   └── heart_cleaning.sql    # SQL cleaning workflow (DuckDB)
├── src/
│   ├── preprocess.py         # Data preprocessing pipeline
│   ├── train_model.py        # ML training and evaluation
│   └── generate_model_brief.py # Build summary from metrics
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

## Data privacy / Git safety

- Put UCI files in `data/raw/` locally for running the pipeline.
- `data/raw/*` should stay untracked so private/local datasets are never pushed.
- Keep only `data/raw/.gitkeep` in GitHub.


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

## Step 3: Generate decision-support narrative

```bash
python src/generate_model_brief.py
```

Output:
- `reports/model_brief.md`

You can share `reports/model_brief.md` with stakeholders to generate:
- model comparison narrative,
- deployment recommendation,
- risk/ethics caveats,
- next experiments for improvement.

## Why this is strong for your job profile

You demonstrate:
- **Python data engineering** (cleaning and feature prep)
- **SQL transformation skills**
- **ML modeling and evaluation**
- **AI-assisted interpretability and communication**


## How to put this into your GitHub repo

If your repo already exists on GitHub, use these commands from your project root:

```bash
# 1) Initialize git only if needed
git init

# 2) Add your GitHub repo as remote (replace with your URL)
git remote add origin https://github.com/<your-username>/<your-repo>.git

# If origin already exists, update it instead:
# git remote set-url origin https://github.com/<your-username>/<your-repo>.git

# 3) Commit files
git add .
git commit -m "Add heart disease ML+AI pipeline"

# 4) Push to GitHub
git branch -M main
git push -u origin main
```

If you already have commits and just want to push updates:

```bash
git add .
git commit -m "Update preprocessing/training workflow"
git push
```

### Recommended commit flow for this project

1. `feat: preprocessing pipeline`
2. `feat: SQL cleaning workflow`
3. `feat: model training + metrics`
4. `feat: model brief generation`
5. `docs: README with run instructions`

That sequence makes your contribution history easy for recruiters to review.
