-- DuckDB SQL workflow for showcasing SQL-based preprocessing.
-- Run with:
--   duckdb -c ".read sql/heart_cleaning.sql"

CREATE SCHEMA IF NOT EXISTS heart;

CREATE OR REPLACE TABLE heart.raw AS
SELECT
    column0 AS age,
    column1 AS sex,
    column2 AS cp,
    column3 AS trestbps,
    column4 AS chol,
    column5 AS fbs,
    column6 AS restecg,
    column7 AS thalach,
    column8 AS exang,
    column9 AS oldpeak,
    column10 AS slope,
    column11 AS ca,
    column12 AS thal,
    column13 AS num,
    'processed.cleveland.data' AS source_file
FROM read_csv(
    'data/raw/processed.cleveland.data',
    delim=',',
    header=false,
    nullstr='?'
);

-- Add other files when available.
INSERT INTO heart.raw
SELECT
    column0, column1, column2, column3, column4, column5, column6,
    column7, column8, column9, column10, column11, column12, column13,
    'processed.hungarian.data'
FROM read_csv(
    'data/raw/processed.hungarian.data',
    delim=',',
    header=false,
    nullstr='?'
);

INSERT INTO heart.raw
SELECT
    column0, column1, column2, column3, column4, column5, column6,
    column7, column8, column9, column10, column11, column12, column13,
    'processed.switzerland.data'
FROM read_csv(
    'data/raw/processed.switzerland.data',
    delim=',',
    header=false,
    nullstr='?'
);

INSERT INTO heart.raw
SELECT
    column0, column1, column2, column3, column4, column5, column6,
    column7, column8, column9, column10, column11, column12, column13,
    'processed.va.data'
FROM read_csv(
    'data/raw/processed.va.data',
    delim=',',
    header=false,
    nullstr='?'
);

CREATE OR REPLACE TABLE heart.cleaned AS
SELECT
    TRY_CAST(age AS INTEGER) AS age,
    TRY_CAST(sex AS INTEGER) AS sex,
    TRY_CAST(cp AS INTEGER) AS cp,
    TRY_CAST(trestbps AS INTEGER) AS trestbps,
    TRY_CAST(chol AS INTEGER) AS chol,
    TRY_CAST(fbs AS INTEGER) AS fbs,
    TRY_CAST(restecg AS INTEGER) AS restecg,
    TRY_CAST(thalach AS INTEGER) AS thalach,
    TRY_CAST(exang AS INTEGER) AS exang,
    TRY_CAST(oldpeak AS DOUBLE) AS oldpeak,
    TRY_CAST(slope AS INTEGER) AS slope,
    TRY_CAST(ca AS INTEGER) AS ca,
    TRY_CAST(thal AS INTEGER) AS thal,
    TRY_CAST(num AS INTEGER) AS num,
    CASE WHEN TRY_CAST(num AS INTEGER) > 0 THEN 1 ELSE 0 END AS target,
    source_file
FROM heart.raw;

COPY heart.cleaned TO 'data/processed/heart_cleaned_sql.csv' (HEADER, DELIMITER ',');

