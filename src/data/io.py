from __future__ import annotations

from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".parquet"}


def load_table(path: str | Path) -> pd.DataFrame:
    """Загружает таблицу CSV/Parquet по расширению."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Неподдерживаемый формат '{suffix}'. Ожидаются: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def load_first_supported_file(raw_dir: str | Path) -> tuple[Path, pd.DataFrame]:
    """Находит первый CSV/Parquet в каталоге и загружает его."""
    raw_dir = Path(raw_dir)
    candidates = sorted(
        p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError(f"В каталоге {raw_dir} нет CSV/Parquet файлов")

    first = candidates[0]
    return first, load_table(first)
