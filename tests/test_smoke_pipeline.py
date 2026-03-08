from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.io import load_table
from src.data.validation import validate_frame
from src.features.preprocessing import split_features_target
from src.models.pipeline import train_baseline


def build_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num_feature": [1.0, 2.5, 3.2, None, 5.1, 6.2],
            "cat_feature": ["a", "b", "a", "c", None, "b"],
            "target": [10.0, 12.0, 11.5, 9.0, 13.2, 12.7],
        }
    )


def test_csv_and_parquet_loader(tmp_path: Path) -> None:
    df = build_df()
    csv_path = tmp_path / "sample.csv"
    parquet_path = tmp_path / "sample.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    loaded_csv = load_table(csv_path)
    loaded_parquet = load_table(parquet_path)

    assert loaded_csv.shape == df.shape
    assert loaded_parquet.shape == df.shape


def test_train_baseline_smoke() -> None:
    df = build_df()
    report = validate_frame(df)
    assert report.rows == 6

    split = split_features_target(df, "target")
    result = train_baseline(split.X, split.y, test_size=0.33, random_state=42)

    assert "rmse" in result.metrics
    assert result.metrics["n_train"] > 0
