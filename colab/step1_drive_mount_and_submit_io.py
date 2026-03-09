"""Colab Cell 1: монтирование Google Drive + подготовка путей.

Запуск в Colab (1-я ячейка):
    %run /content/ML-1-task-08-03-2026/colab/step1_drive_mount_and_submit_io.py

После запуска в globals появится словарь COLAB_PATHS:
- COLAB_PATHS['data_dir']
- COLAB_PATHS['submissions_dir']
- COLAB_PATHS['sample_submit_path']
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from zipfile import ZipFile

import pandas as pd

EXPECTED_PARQUET_FILES = {
    "pretrain_part_1.parquet",
    "pretrain_part_2.parquet",
    "pretrain_part_3.parquet",
    "train_part_1.parquet",
    "train_part_2.parquet",
    "train_part_3.parquet",
    "pretest.parquet",
    "test.parquet",
    "train_labels.parquet",
}


def mount_google_drive(mount_point: str = "/content/drive") -> None:
    from google.colab import drive  # type: ignore

    drive.mount(mount_point, force_remount=False)


def _missing_expected_files(data_dir: Path):
    existing = {p.name for p in data_dir.glob("*.parquet")}
    return sorted(EXPECTED_PARQUET_FILES - existing)


def _extract_zip_if_needed(labels_zip_path: Path, data_dir: Path) -> None:
    missing_before = _missing_expected_files(data_dir)
    if not missing_before:
        return

    if not labels_zip_path.exists():
        raise FileNotFoundError(
            f"Не найден zip: {labels_zip_path}. "
            "Для вашего кейса ожидается /content/drive/MyDrive/data/train_labels.zip"
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(labels_zip_path, "r") as zf:
        zf.extractall(data_dir)

    missing_after = _missing_expected_files(data_dir)
    if missing_after:
        raise RuntimeError("После распаковки не найдены файлы: " + ", ".join(missing_after))


def load_sample_submit(sample_submit_path: str) -> pd.DataFrame:
    sample = pd.read_csv(sample_submit_path)
    need = {"event_id", "predict"}
    miss = need - set(sample.columns)
    if miss:
        raise ValueError(f"sample_submit.csv не содержит колонки: {sorted(miss)}")
    return sample


def save_versioned_submission(
    predictions: pd.DataFrame,
    submissions_dir: str,
    run_name: str,
    sample_submit_path: Optional[str] = None,
) -> str:
    need = {"event_id", "predict"}
    miss = need - set(predictions.columns)
    if miss:
        raise ValueError(f"В predictions отсутствуют колонки: {sorted(miss)}")

    submit_df = predictions[["event_id", "predict"]].copy()
    if sample_submit_path:
        sample = load_sample_submit(sample_submit_path)
        if len(submit_df) != len(sample):
            raise ValueError(f"Длина submit ({len(submit_df)}) != sample ({len(sample)})")
        if set(submit_df["event_id"].astype(str)) != set(sample["event_id"].astype(str)):
            raise ValueError("event_id в submit не совпадают с sample_submit")

    out_dir = Path(submissions_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"submit__{run_name}__{ts}.csv"
    submit_df.to_csv(out_path, index=False)
    return str(out_path)




def resolve_sample_submit_path(preferred_path: str = "") -> str:
    candidates = []
    if preferred_path:
        candidates.append(Path(preferred_path))

    candidates.extend([
        Path('/content/drive/MyDrive/data/sample_submit.csv'),
        Path('/content/drive/MyDrive/sample_submit.csv'),
        Path('/content/ML-1-task-08-03-2026/sample_submit.csv'),
    ])

    for c in candidates:
        if c.exists():
            return str(c)

    raise FileNotFoundError(
        "Не найден sample_submit.csv. Проверьте один из путей: "
        + ", ".join(str(c) for c in candidates)
    )

def setup_colab_paths(
    data_dir: str = "/content/drive/MyDrive/data/raw",
    labels_zip_path: str = "/content/drive/MyDrive/data/train_labels.zip",
    submissions_dir_name: str = "submissions_strazh",
    sample_submit_path: str = "",
) -> Dict[str, str]:
    data_dir_p = Path(data_dir)
    data_dir_p.mkdir(parents=True, exist_ok=True)

    _extract_zip_if_needed(Path(labels_zip_path), data_dir_p)
    missing = _missing_expected_files(data_dir_p)
    if missing:
        raise RuntimeError("В data_dir нет обязательных parquet: " + ", ".join(missing))

    submissions_dir = data_dir_p.parent / submissions_dir_name
    submissions_dir.mkdir(parents=True, exist_ok=True)

    resolved_sample_submit = resolve_sample_submit_path(sample_submit_path)

    out = {
        "data_dir": str(data_dir_p),
        "labels_zip_path": labels_zip_path,
        "submissions_dir": str(submissions_dir),
        "sample_submit_path": resolved_sample_submit,
    }
    return out


if __name__ == "__main__":
    print("[STEP1] Монтируем Google Drive...")
    mount_google_drive("/content/drive")

    print("[STEP1] Готовим пути и проверяем parquet...")
    COLAB_PATHS = setup_colab_paths()
    print("[STEP1] Готово. Используйте COLAB_PATHS во 2-й ячейке:")
    print(COLAB_PATHS)
    print("[STEP1] sample_submit:", COLAB_PATHS["sample_submit_path"])
