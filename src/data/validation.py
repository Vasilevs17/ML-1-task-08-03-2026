from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ValidationReport:
    rows: int
    cols: int
    duplicated_rows: int
    total_missing: int


def validate_frame(df: pd.DataFrame) -> ValidationReport:
    if df.empty:
        raise ValueError("Датасет пустой")

    return ValidationReport(
        rows=len(df),
        cols=df.shape[1],
        duplicated_rows=int(df.duplicated().sum()),
        total_missing=int(df.isna().sum().sum()),
    )
