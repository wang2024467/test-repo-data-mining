from pathlib import Path


def test_project_structure_exists() -> None:
    assert Path('src/preprocess.py').exists()
    assert Path('src/train_model.py').exists()
    assert Path('src/generate_llm_brief.py').exists()
    assert Path('sql/heart_cleaning.sql').exists()
