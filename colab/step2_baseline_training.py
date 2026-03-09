"""Colab Cell 2: baseline обучение + PR-AUC + сохранение submit (memory-safe).

Запуск в Colab (2-я ячейка, после step1):
    %run /content/ML-1-task-08-03-2026/colab/step2_baseline_training.py

Особенности этой версии:
- потоковая загрузка train_part_* батчами (без чтения целых файлов в RAM);
- в обучение попадают только размеченные пары (customer_id, event_id);
- подробные логи по этапам, времени и памяти (если доступен psutil).
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


START_TS = time.time()


def _fmt_elapsed(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _mem_gb() -> str:
    if psutil is None:
        return "n/a"
    rss = psutil.Process().memory_info().rss / (1024**3)
    return f"{rss:.2f} GB"


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    elapsed = _fmt_elapsed(time.time() - START_TS)
    print(f"[{now} | +{elapsed} | RAM { _mem_gb() }] {msg}")


def _ram_guard(limit_gb: float = 11.5) -> None:
    if psutil is None:
        return
    rss = psutil.Process().memory_info().rss / (1024**3)
    if rss > limit_gb:
        raise MemoryError(
            f"Текущая RAM {rss:.2f} GB > лимита {limit_gb:.2f} GB. "
            "Уменьшите batch_size (например, до 100_000)."
        )


def _read_parquet(path: Path, columns: List[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def _filter_labeled_rows_streaming(
    parquet_path: Path,
    labels_key_df: pd.DataFrame,
    batch_size: int = 200_000,
) -> pd.DataFrame:
    """Читает parquet потоково и оставляет только строки из labels по (customer_id,event_id)."""
    pf = pq.ParquetFile(parquet_path)
    out_parts: List[pd.DataFrame] = []

    labels_index = pd.MultiIndex.from_frame(labels_key_df[["customer_id", "event_id"]])

    total_rows = pf.metadata.num_rows
    processed = 0
    matched_rows = 0

    for batch in pf.iter_batches(batch_size=batch_size):
        df_batch = batch.to_pandas()
        bidx = pd.MultiIndex.from_frame(df_batch[["customer_id", "event_id"]])
        mask = bidx.isin(labels_index)
        if mask.any():
            keep = df_batch.loc[mask].copy()
            out_parts.append(keep)
            matched_rows += int(mask.sum())

        processed += len(df_batch)
        if processed % (batch_size * 10) == 0 or processed == total_rows:
            pct = processed / max(total_rows, 1) * 100
            log(f"{parquet_path.name}: обработано {processed:,}/{total_rows:,} ({pct:.1f}%), матчей {matched_rows:,}")
            _ram_guard(11.5)

    if not out_parts:
        return pd.DataFrame(columns=list(labels_key_df.columns))

    return pd.concat(out_parts, ignore_index=True)


def load_labeled_train_memory_safe(data_dir: str, batch_size: int = 200_000) -> pd.DataFrame:
    """Загружает только размеченные события из train_part_* без OOM."""
    base = Path(data_dir)
    labels = _read_parquet(base / "train_labels.parquet", ["customer_id", "event_id", "target"])
    labels_keys = labels[["customer_id", "event_id"]].drop_duplicates().copy()

    log(f"Загружены labels: {len(labels):,} строк")

    train_files = sorted(base.glob("train_part_*.parquet"))
    if not train_files:
        raise FileNotFoundError("Не найдены train_part_*.parquet")

    selected_parts: List[pd.DataFrame] = []
    for p in train_files:
        log(f"Старт потоковой фильтрации {p.name}...")
        filtered = _filter_labeled_rows_streaming(p, labels_keys, batch_size=batch_size)
        if not filtered.empty:
            selected_parts.append(filtered)
        log(f"Финиш {p.name}: взято {len(filtered):,} строк")

    if not selected_parts:
        raise RuntimeError("Не найдено ни одной размеченной строки в train_part_*")

    train_df = pd.concat(selected_parts, ignore_index=True)
    train_df = train_df.merge(labels, on=["customer_id", "event_id"], how="inner")
    train_df = train_df.drop_duplicates(subset=["customer_id", "event_id"]).copy()

    log(f"Итоговый labeled train: {len(train_df):,} строк")
    return train_df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["event_dttm"] = pd.to_datetime(out["event_dttm"], errors="coerce")
    out["event_hour"] = out["event_dttm"].dt.hour.astype("float32")
    out["event_dayofweek"] = out["event_dttm"].dt.dayofweek.astype("float32")
    out["event_day"] = out["event_dttm"].dt.day.astype("float32")
    out["event_month"] = out["event_dttm"].dt.month.astype("float32")
    out["is_weekend"] = (out["event_dayofweek"] >= 5).astype("float32")
    return out


def make_time_aware_split(
    df: pd.DataFrame,
    target_col: str = "target",
    valid_share: float = 0.2,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Timestamp]:
    w = df.copy()
    w["event_dttm"] = pd.to_datetime(w["event_dttm"], errors="coerce")
    if w["event_dttm"].isna().all():
        raise RuntimeError("event_dttm не парсится в datetime")

    cutoff = w["event_dttm"].quantile(1.0 - valid_share)
    tr_mask = w["event_dttm"] < cutoff
    va_mask = ~tr_mask

    y = w[target_col].astype("int8")
    x_train = w.loc[tr_mask].drop(columns=[target_col])
    y_train = y.loc[tr_mask]
    x_valid = w.loc[va_mask].drop(columns=[target_col])
    y_valid = y.loc[va_mask]

    log(f"Time split: train={len(x_train):,}, valid={len(x_valid):,}, cutoff={cutoff}")
    return x_train, y_train, x_valid, y_valid, cutoff


def _build_model(feature_df: pd.DataFrame) -> Pipeline:
    excluded = {"customer_id", "event_id", "event_dttm"}
    fcols = [c for c in feature_df.columns if c not in excluded]
    num_cols = [c for c in fcols if pd.api.types.is_numeric_dtype(feature_df[c])]
    cat_cols = [c for c in fcols if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=50)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        [
            ("pre", pre),
            (
                "clf",
                LogisticRegression(
                    max_iter=250,
                    class_weight="balanced",
                    solver="saga",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    return model




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

def save_versioned_submission(
    predictions: pd.DataFrame,
    submissions_dir: str,
    run_name: str,
    sample_submit_path: str,
) -> str:
    sample = pd.read_csv(sample_submit_path)
    need = {"event_id", "predict"}
    if need - set(predictions.columns):
        raise ValueError("predictions должен иметь колонки event_id,predict")
    if need - set(sample.columns):
        raise ValueError("sample_submit.csv должен иметь колонки event_id,predict")

    sub = predictions[["event_id", "predict"]].copy()
    if len(sub) != len(sample):
        raise ValueError("Размер submit не совпадает с sample_submit")
    if set(sub["event_id"].astype(str)) != set(sample["event_id"].astype(str)):
        raise ValueError("event_id submit не совпадают с sample_submit")

    out_dir = Path(submissions_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"submit__{run_name}__{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(out_path, index=False)
    return str(out_path)


def run_baseline(
    data_dir: str,
    sample_submit_path: str,
    submissions_dir: str,
    run_name: str = "baseline_v1",
    batch_size: int = 200_000,
) -> Dict[str, object]:
    log("Шаг 1/6: чтение размеченного train в memory-safe режиме")
    train_df = load_labeled_train_memory_safe(data_dir, batch_size=batch_size)

    log("Шаг 2/6: генерация time features")
    train_df = add_time_features(train_df)

    log("Шаг 3/6: time-aware split")
    x_train, y_train, x_valid, y_valid, cutoff = make_time_aware_split(train_df, valid_share=0.2)

    log("Шаг 4/6: обучение baseline модели")
    model = _build_model(x_train)
    _ram_guard(11.5)
    model.fit(x_train, y_train)

    valid_pred = model.predict_proba(x_valid)[:, 1]
    pr_auc = float(average_precision_score(y_valid, valid_pred))
    log(f"Валидация завершена. PR-AUC={pr_auc:.6f}")

    log("Шаг 5/6: финальное обучение на полном labeled train")
    full_x = train_df.drop(columns=["target"])
    full_y = train_df["target"].astype("int8")
    final_model = _build_model(full_x)
    _ram_guard(11.5)
    final_model.fit(full_x, full_y)

    log("Шаг 6/6: предсказание test и сохранение submit")
    test_df = add_time_features(_read_parquet(Path(data_dir) / "test.parquet"))
    test_pred = final_model.predict_proba(test_df)[:, 1]
    submit = pd.DataFrame({"event_id": test_df["event_id"].values, "predict": test_pred.astype(np.float64)})

    sample_submit_path = resolve_sample_submit_path(sample_submit_path)
    log(f"Используем sample_submit: {sample_submit_path}")

    saved = save_versioned_submission(
        predictions=submit,
        submissions_dir=submissions_dir,
        run_name=run_name,
        sample_submit_path=sample_submit_path,
    )

    return {
        "valid_pr_auc": pr_auc,
        "train_size": int(len(x_train)),
        "valid_size": int(len(x_valid)),
        "cutoff_ts": str(cutoff),
        "saved_submit_path": saved,
        "peak_ram_hint": "Следите по логам RAM; целевой предел ~11.5GB",
    }


if __name__ == "__main__":
    defaults = {
        "data_dir": "/content/drive/MyDrive/data/raw",
        "sample_submit_path": "",
        "submissions_dir": "/content/drive/MyDrive/data/submissions_strazh",
    }

    if "COLAB_PATHS" in globals() and isinstance(globals()["COLAB_PATHS"], dict):
        cp = globals()["COLAB_PATHS"]
        defaults.update(
            {
                "data_dir": cp.get("data_dir", defaults["data_dir"]),
                "sample_submit_path": cp.get("sample_submit_path", defaults["sample_submit_path"]),
                "submissions_dir": cp.get("submissions_dir", defaults["submissions_dir"]),
            }
        )

    try:
        resolved = resolve_sample_submit_path(defaults.get("sample_submit_path", ""))
        defaults["sample_submit_path"] = resolved
        log(f"[STEP2] Найден sample_submit: {resolved}")
    except FileNotFoundError as e:
        log(f"[STEP2] ERROR: {e}")
        raise

    log("[STEP2] Старт baseline обучения...")
    result = run_baseline(**defaults, run_name="baseline_v1", batch_size=200_000)
    log("[STEP2] Готово")
    print(result)
