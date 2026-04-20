from pathlib import Path

import pandas as pd

from src.preprocess import COLUMNS, clean_dataframe, encode_for_model


def test_project_structure_exists() -> None:
    assert Path('src/preprocess.py').exists()
    assert Path('src/train_model.py').exists()
    assert Path('src/generate_model_brief.py').exists()
    assert Path('sql/heart_cleaning.sql').exists()


def test_encode_for_model_handles_missing_values() -> None:
    # Includes '?' placeholders that become missing values after cleaning.
    rows = [
        [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],
        [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, '?', 3, 2],
        [67, 1, 4, 120, 229, 0, 2, 129, 1, 2.6, 2, 2, '?', 1],
    ]
    raw_df = pd.DataFrame(rows, columns=COLUMNS)

    cleaned_df = clean_dataframe(raw_df)
    model_df = encode_for_model(cleaned_df)

    # Should not raise and should produce numeric matrix + target column.
    assert 'target' in model_df.columns
    assert len(model_df) == 3
    assert not model_df.isna().any().any()
