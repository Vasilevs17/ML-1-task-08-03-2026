"""
ONE-CELL Colab script: улучшенный baseline для задачи антифрода.

Как использовать в Colab:
1) Скопируйте весь файл в одну ячейку ИЛИ выполните:
   %run /content/ML-1-task-08-03-2026/colab/one_cell_improved_model.py

Что делает скрипт:
- монтирует Google Drive;
- находит sample_submit.csv (включая /content/drive/MyDrive/data/sample_submit.csv);
- memory-safe извлекает размеченный train из train_part_*;
- строит customer-level history фичи из pretrain_part_* + train_part_* + pretest.parquet (стримингом);
- обучает CatBoost (если нет — ставит пакет), time-aware split, считает PR-AUC;
- сохраняет submit__catboost_hist_v1__*.csv рядом с данными на Google Drive.
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# =========================
# Конфиг (при желании меняйте)
# =========================
DATA_DIR = "/content/drive/MyDrive/data/raw"
LABELS_ZIP = "/content/drive/MyDrive/data/train_labels.zip"
SUBMISSIONS_DIR = "/content/drive/MyDrive/data/submissions_strazh"
SAMPLE_SUBMIT_HINT = ""  # можно оставить пустым
RAM_LIMIT_GB = 11.5
BATCH_SIZE_ROWS = 4_000_000
MIN_BATCH_ROWS = 1_000_000
MAX_BATCH_ROWS = 10_000_000
TARGET_RAM_UTIL_GB = 10.0
RANDOM_STATE = 42

# train params
VALID_SHARE = 0.2
CATBOOST_ITERS = 2500
CATBOOST_LR = 0.05
CATBOOST_DEPTH = 8

START_TS = time.time()


def _fmt_elapsed(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _try_import_psutil():
    try:
        import psutil

        return psutil
    except Exception:
        return None


psutil = _try_import_psutil()


def _ram_gb() -> float:
    if psutil is None:
        return -1.0
    return float(psutil.Process().memory_info().rss / (1024**3))


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    elapsed = _fmt_elapsed(time.time() - START_TS)
    ram = _ram_gb()
    ram_s = f"{ram:.2f} GB" if ram >= 0 else "n/a"
    print(f"[{now} | +{elapsed} | RAM {ram_s}] {msg}")


def ram_guard(limit_gb: float = RAM_LIMIT_GB) -> None:
    ram = _ram_gb()
    if ram >= 0 and ram > limit_gb:
        raise MemoryError(
            f"RAM {ram:.2f} GB > лимита {limit_gb:.2f} GB. "
            f"Снизьте BATCH_SIZE_ROWS (сейчас {BATCH_SIZE_ROWS:,})."
        )


def auto_tune_batch_size(default_batch: int = BATCH_SIZE_ROWS) -> int:
    """Агрессивная утилизация RAM, но ниже hard-limit 11.5 GB."""
    ram = _ram_gb()
    if ram < 0:
        return default_batch

    if ram < 2.0:
        tuned = int(default_batch * 2.2)
    elif ram < 4.0:
        tuned = int(default_batch * 1.8)
    elif ram < 6.0:
        tuned = int(default_batch * 1.5)
    elif ram < 8.0:
        tuned = int(default_batch * 1.25)
    else:
        tuned = default_batch

    tuned = max(MIN_BATCH_ROWS, min(MAX_BATCH_ROWS, tuned))
    return tuned


def ensure_catboost_installed() -> None:
    try:
        import catboost  # noqa: F401

        return
    except Exception:
        log("CatBoost не найден, устанавливаю...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost"])


def mount_drive() -> None:
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive", force_remount=False)


def resolve_sample_submit_path(preferred: str = "") -> str:
    candidates: List[Path] = []
    if preferred:
        candidates.append(Path(preferred))
    candidates.extend(
        [
            Path("/content/drive/MyDrive/data/sample_submit.csv"),
            Path("/content/drive/MyDrive/sample_submit.csv"),
            Path("/content/ML-1-task-08-03-2026/sample_submit.csv"),
        ]
    )
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError("Не найден sample_submit.csv. Проверил: " + ", ".join(map(str, candidates)))


def ensure_data_present(data_dir: Path, labels_zip: Path) -> None:
    expected = {
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
    data_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in data_dir.glob("*.parquet")}
    missing = sorted(expected - existing)
    if not missing:
        return

    if not labels_zip.exists():
        raise FileNotFoundError(
            f"Не хватает parquet ({missing}), и не найден zip: {labels_zip}."
        )
    log(f"Не хватает {len(missing)} parquet, распаковываю zip...")
    import zipfile

    with zipfile.ZipFile(labels_zip, "r") as zf:
        zf.extractall(data_dir)

    existing = {p.name for p in data_dir.glob("*.parquet")}
    missing = sorted(expected - existing)
    if missing:
        raise RuntimeError("После распаковки все еще не хватает: " + ", ".join(missing))


def load_labels(data_dir: Path) -> pd.DataFrame:
    labels = pd.read_parquet(data_dir / "train_labels.parquet", columns=["customer_id", "event_id", "target"])
    labels["customer_id"] = labels["customer_id"].astype("int64")
    labels["event_id"] = labels["event_id"].astype("int64")
    labels["target"] = labels["target"].astype("int8")
    return labels


def stream_filter_labeled_rows(parquet_path: Path, label_keys: pd.MultiIndex, batch_size: int) -> pd.DataFrame:
    pf = pq.ParquetFile(parquet_path)
    rows_total = pf.metadata.num_rows
    taken_parts: List[pd.DataFrame] = []
    processed = 0
    matched = 0

    for rb in pf.iter_batches(batch_size=batch_size):
        batch = rb.to_pandas()
        idx = pd.MultiIndex.from_frame(batch[["customer_id", "event_id"]])
        mask = idx.isin(label_keys)
        if mask.any():
            taken = batch.loc[mask].copy()
            taken_parts.append(taken)
            matched += int(mask.sum())

        processed += len(batch)
        if processed % (batch_size * 5) == 0 or processed == rows_total:
            pct = 100.0 * processed / max(rows_total, 1)
            log(f"{parquet_path.name}: {processed:,}/{rows_total:,} ({pct:.1f}%), matched={matched:,}")
            ram_guard()

    if not taken_parts:
        return pd.DataFrame()
    out = pd.concat(taken_parts, ignore_index=True)
    del taken_parts
    gc.collect()
    return out


def build_labeled_train(data_dir: Path, labels: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    key_mi = pd.MultiIndex.from_frame(labels[["customer_id", "event_id"]])

    selected = []
    for p in sorted(data_dir.glob("train_part_*.parquet")):
        log(f"Фильтрую размеченные строки из {p.name}...")
        part = stream_filter_labeled_rows(p, key_mi, batch_size=batch_size)
        if not part.empty:
            selected.append(part)
        log(f"{p.name}: взято {len(part):,} строк")
        del part
        gc.collect()
        ram_guard()

    if not selected:
        raise RuntimeError("Не удалось извлечь размеченные строки из train_part_*")

    train = pd.concat(selected, ignore_index=True)
    del selected
    gc.collect()

    train = train.merge(labels, on=["customer_id", "event_id"], how="inner")
    train = train.drop_duplicates(["customer_id", "event_id"]).copy()

    log(f"Labeled train сформирован: {len(train):,} строк")
    return train


@dataclass
class CustAggState:
    cnt: Dict[int, int]
    amt_sum: Dict[int, float]
    amt_sumsq: Dict[int, float]
    amt_min: Dict[int, float]
    amt_max: Dict[int, float]
    hour_sum: Dict[int, float]
    weekend_sum: Dict[int, float]


def _new_state() -> CustAggState:
    return CustAggState(cnt={}, amt_sum={}, amt_sumsq={}, amt_min={}, amt_max={}, hour_sum={}, weekend_sum={})


def _upd_dict_sum(dst: Dict[int, float], s: pd.Series) -> None:
    for k, v in s.items():
        dst[int(k)] = dst.get(int(k), 0.0) + float(v)


def _upd_dict_min(dst: Dict[int, float], s: pd.Series) -> None:
    for k, v in s.items():
        kk = int(k)
        vv = float(v)
        if kk not in dst or vv < dst[kk]:
            dst[kk] = vv


def _upd_dict_max(dst: Dict[int, float], s: pd.Series) -> None:
    for k, v in s.items():
        kk = int(k)
        vv = float(v)
        if kk not in dst or vv > dst[kk]:
            dst[kk] = vv


def update_customer_state_from_batch(state: CustAggState, df: pd.DataFrame) -> None:
    if df.empty:
        return

    df = df[["customer_id", "operaton_amt", "event_dttm"]].copy()
    df["event_dttm"] = pd.to_datetime(df["event_dttm"], errors="coerce")
    df["event_hour"] = df["event_dttm"].dt.hour.astype("float32")
    df["is_weekend"] = (df["event_dttm"].dt.dayofweek >= 5).astype("float32")
    df["operaton_amt"] = pd.to_numeric(df["operaton_amt"], errors="coerce")

    g_cnt = df.groupby("customer_id", dropna=False).size()
    _upd_dict_sum(state.cnt, g_cnt.astype("float64"))

    amt = df.dropna(subset=["operaton_amt"])
    if not amt.empty:
        g_sum = amt.groupby("customer_id", dropna=False)["operaton_amt"].sum()
        g_sumsq = amt.assign(_sq=amt["operaton_amt"] ** 2).groupby("customer_id", dropna=False)["_sq"].sum()
        g_min = amt.groupby("customer_id", dropna=False)["operaton_amt"].min()
        g_max = amt.groupby("customer_id", dropna=False)["operaton_amt"].max()
        _upd_dict_sum(state.amt_sum, g_sum)
        _upd_dict_sum(state.amt_sumsq, g_sumsq)
        _upd_dict_min(state.amt_min, g_min)
        _upd_dict_max(state.amt_max, g_max)

    g_hour = df.dropna(subset=["event_hour"]).groupby("customer_id", dropna=False)["event_hour"].sum()
    g_week = df.dropna(subset=["is_weekend"]).groupby("customer_id", dropna=False)["is_weekend"].sum()
    _upd_dict_sum(state.hour_sum, g_hour)
    _upd_dict_sum(state.weekend_sum, g_week)


def build_customer_history_features(data_dir: Path, batch_size: int) -> pd.DataFrame:
    files = [
        data_dir / "pretrain_part_1.parquet",
        data_dir / "pretrain_part_2.parquet",
        data_dir / "pretrain_part_3.parquet",
        data_dir / "train_part_1.parquet",
        data_dir / "train_part_2.parquet",
        data_dir / "train_part_3.parquet",
        data_dir / "pretest.parquet",
    ]

    state = _new_state()

    for fp in files:
        if not fp.exists():
            continue
        log(f"История: сканирую {fp.name}...")
        pf = pq.ParquetFile(fp)
        total = pf.metadata.num_rows
        processed = 0

        for rb in pf.iter_batches(batch_size=batch_size, columns=["customer_id", "operaton_amt", "event_dttm"]):
            b = rb.to_pandas()
            update_customer_state_from_batch(state, b)
            processed += len(b)
            if processed % (batch_size * 5) == 0 or processed == total:
                pct = 100.0 * processed / max(total, 1)
                log(f"{fp.name}: {processed:,}/{total:,} ({pct:.1f}%)")
                ram_guard()
            del b
        gc.collect()

    cust_ids = sorted(state.cnt.keys())
    hist = pd.DataFrame({"customer_id": cust_ids})
    hist["cust_txn_cnt"] = hist["customer_id"].map(lambda x: float(state.cnt.get(int(x), 0.0))).astype("float32")
    hist["cust_amt_sum"] = hist["customer_id"].map(lambda x: float(state.amt_sum.get(int(x), 0.0))).astype("float32")
    hist["cust_amt_sumsq"] = hist["customer_id"].map(lambda x: float(state.amt_sumsq.get(int(x), 0.0))).astype("float32")
    hist["cust_amt_min"] = hist["customer_id"].map(lambda x: state.amt_min.get(int(x), np.nan)).astype("float32")
    hist["cust_amt_max"] = hist["customer_id"].map(lambda x: state.amt_max.get(int(x), np.nan)).astype("float32")
    hist["cust_hour_sum"] = hist["customer_id"].map(lambda x: float(state.hour_sum.get(int(x), 0.0))).astype("float32")
    hist["cust_weekend_sum"] = hist["customer_id"].map(lambda x: float(state.weekend_sum.get(int(x), 0.0))).astype("float32")

    denom = hist["cust_txn_cnt"].replace(0, np.nan)
    hist["cust_amt_mean"] = (hist["cust_amt_sum"] / denom).astype("float32")
    var = (hist["cust_amt_sumsq"] / denom) - (hist["cust_amt_mean"] ** 2)
    hist["cust_amt_std"] = np.sqrt(np.clip(var, 0, None)).astype("float32")
    hist["cust_hour_mean"] = (hist["cust_hour_sum"] / denom).astype("float32")
    hist["cust_weekend_rate"] = (hist["cust_weekend_sum"] / denom).astype("float32")

    keep_cols = [
        "customer_id",
        "cust_txn_cnt",
        "cust_amt_sum",
        "cust_amt_min",
        "cust_amt_max",
        "cust_amt_mean",
        "cust_amt_std",
        "cust_hour_mean",
        "cust_weekend_rate",
    ]
    hist = hist[keep_cols]
    log(f"History features готовы: {hist.shape}")
    return hist


def add_row_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["event_dttm"] = pd.to_datetime(out["event_dttm"], errors="coerce")
    out["event_hour"] = out["event_dttm"].dt.hour.astype("float32")
    out["event_dayofweek"] = out["event_dttm"].dt.dayofweek.astype("float32")
    out["event_day"] = out["event_dttm"].dt.day.astype("float32")
    out["event_month"] = out["event_dttm"].dt.month.astype("float32")
    out["is_weekend"] = (out["event_dayofweek"] >= 5).astype("float32")
    out["operaton_amt"] = pd.to_numeric(out["operaton_amt"], errors="coerce").astype("float32")
    return out


def time_aware_split(df: pd.DataFrame, target_col: str = "target", valid_share: float = VALID_SHARE):
    w = df.copy()
    w["event_dttm"] = pd.to_datetime(w["event_dttm"], errors="coerce")
    cutoff = w["event_dttm"].quantile(1.0 - valid_share)
    tr = w[w["event_dttm"] < cutoff].copy()
    va = w[w["event_dttm"] >= cutoff].copy()
    return tr.drop(columns=[target_col]), tr[target_col].astype("int8"), va.drop(columns=[target_col]), va[target_col].astype("int8"), cutoff


def pick_feature_columns(df: pd.DataFrame, target_col: Optional[str]) -> List[str]:
    drop = {"event_id", "event_dttm", "customer_id"}
    if target_col:
        drop.add(target_col)
    return [c for c in df.columns if c not in drop]


def prepare_for_catboost(df: pd.DataFrame, feature_cols: List[str], cat_cols: List[str]) -> pd.DataFrame:
    out = df[feature_cols].copy()
    out = out.replace({None: np.nan})

    # Доп. страховка: все object/string/category приводим к str, даже если не попали в cat_cols.
    obj_like = [c for c in out.columns if str(out[c].dtype) in ("object", "string", "category")]
    for c in set(cat_cols).union(obj_like):
        out[c] = out[c].astype("string").fillna("__MISSING__").astype(str)
        out[c] = out[c].replace({"<NA>": "__MISSING__", "nan": "__MISSING__", "None": "__MISSING__"})

    # Числовые: float32 + fillna.
    num_cols = [c for c in feature_cols if c not in set(cat_cols).union(set(obj_like))]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
        med = out[c].median()
        if pd.isna(med):
            med = 0.0
        out[c] = out[c].fillna(float(med))

    return out


def fit_catboost(X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series):
    ensure_catboost_installed()
    from catboost import CatBoostClassifier

    feat_cols = [c for c in X_train.columns if c not in ["event_id", "event_dttm", "customer_id"]]
    cat_cols = [c for c in feat_cols if str(X_train[c].dtype) in ("object", "string", "category")]

    Xtr = prepare_for_catboost(X_train, feat_cols, cat_cols)
    Xva = prepare_for_catboost(X_valid, feat_cols, cat_cols)

    pos = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)
    log(f"CatBoost: features={len(feat_cols)}, cat_features={len(cat_cols)}, scale_pos_weight={pos:.3f}")

    log(f"Подготовка матриц CatBoost: Xtr={Xtr.shape}, Xva={Xva.shape}")

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        iterations=CATBOOST_ITERS,
        learning_rate=CATBOOST_LR,
        depth=CATBOOST_DEPTH,
        l2_leaf_reg=5.0,
        random_strength=1.0,
        subsample=0.8,
        bootstrap_type="Bernoulli",
        scale_pos_weight=pos,
        random_seed=RANDOM_STATE,
        verbose=200,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=300,
    )

    model.fit(
        Xtr,
        y_train,
        eval_set=(Xva, y_valid),
        cat_features=cat_cols,
        use_best_model=True,
    )
    return model, feat_cols


def average_precision(y_true: pd.Series, y_score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y_true, y_score))


def save_submission(pred_df: pd.DataFrame, sample_submit_path: str, out_dir: Path, run_name: str) -> str:
    sample = pd.read_csv(sample_submit_path)
    req = {"event_id", "predict"}
    if req - set(sample.columns):
        raise ValueError("В sample_submit нет event_id,predict")

    sub = pred_df[["event_id", "predict"]].copy()
    if len(sub) != len(sample):
        raise ValueError(f"Размер submit {len(sub)} != sample {len(sample)}")

    if set(sub["event_id"].astype(str)) != set(sample["event_id"].astype(str)):
        raise ValueError("Множество event_id в submit не совпадает с sample_submit")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"submit__{run_name}__{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(out_path, index=False)
    return str(out_path)


def main() -> None:
    log("Шаг 0: mount Google Drive")
    mount_drive()

    data_dir = Path(DATA_DIR)
    ensure_data_present(data_dir, Path(LABELS_ZIP))
    sample_submit_path = resolve_sample_submit_path(SAMPLE_SUBMIT_HINT)
    log(f"sample_submit: {sample_submit_path}")

    log("Шаг 1: читаю labels")
    labels = load_labels(data_dir)
    log(f"labels: {labels.shape}, target mean={labels['target'].mean():.5f}")

    tuned_batch = auto_tune_batch_size(BATCH_SIZE_ROWS)
    log(f"Шаг 2: строю labeled train (streaming), batch_size={tuned_batch:,}, target_ram≈{TARGET_RAM_UTIL_GB}GB")
    train = build_labeled_train(data_dir, labels, batch_size=tuned_batch)
    del labels
    gc.collect()
    ram_guard()

    # Ещё раз адаптируем batch перед самым тяжёлым этапом истории
    tuned_batch_hist = auto_tune_batch_size(max(tuned_batch, BATCH_SIZE_ROWS))
    log(f"Шаг 3: строю customer history features (pretrain+train+pretest), batch_size={tuned_batch_hist:,}")
    hist = build_customer_history_features(data_dir, batch_size=tuned_batch_hist)

    log("Шаг 4: row features + merge history")
    train = add_row_features(train)
    train = train.merge(hist, on="customer_id", how="left")

    # test
    test = pd.read_parquet(data_dir / "test.parquet")
    test = add_row_features(test)
    test = test.merge(hist, on="customer_id", how="left")

    del hist
    gc.collect()
    ram_guard()

    # time-aware valid
    log("Шаг 5: time-aware split")
    Xtr, ytr, Xva, yva, cutoff = time_aware_split(train, target_col="target", valid_share=VALID_SHARE)
    log(f"split cutoff={cutoff}, train={len(Xtr):,}, valid={len(Xva):,}")

    log("Шаг 6: обучение CatBoost")
    model, feat_cols = fit_catboost(Xtr, ytr, Xva, yva)

    cat_cols_valid = [c for c in feat_cols if str(Xtr[c].dtype) in ("object", "string", "category")]
    Xva_prepared = prepare_for_catboost(Xva, feat_cols, cat_cols_valid)
    va_pred = model.predict_proba(Xva_prepared)[:, 1]
    val_ap = average_precision(yva, va_pred)
    log(f"VALID PR-AUC={val_ap:.6f}")
    cur_ram = _ram_gb()
    if cur_ram >= 0 and cur_ram < 4.5:
        log("RAM использовалась слабо (<4.5GB). Для более агрессивного режима увеличьте BATCH_SIZE_ROWS или MAX_BATCH_ROWS.")

    log("Шаг 7: дообучение на полном train")
    full_X = train[feat_cols].copy()
    full_y = train["target"].astype("int8")

    ensure_catboost_installed()
    from catboost import CatBoostClassifier

    pos = float((full_y == 0).sum()) / max(float((full_y == 1).sum()), 1.0)
    best_it = int(model.get_best_iteration()) if hasattr(model, "get_best_iteration") and model.get_best_iteration() is not None else 600
    final_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        iterations=max(900, int((best_it + 1) * 1.15)),
        learning_rate=CATBOOST_LR,
        depth=CATBOOST_DEPTH,
        l2_leaf_reg=5.0,
        random_strength=1.0,
        subsample=0.8,
        bootstrap_type="Bernoulli",
        scale_pos_weight=pos,
        random_seed=RANDOM_STATE,
        verbose=200,
        allow_writing_files=False,
    )

    cat_cols = [c for c in feat_cols if str(full_X[c].dtype) in ("object", "string", "category")]
    full_X_prepared = prepare_for_catboost(train, feat_cols, cat_cols)
    test_prepared = prepare_for_catboost(test, feat_cols, cat_cols)

    final_model.fit(full_X_prepared, full_y, cat_features=cat_cols)

    log("Шаг 8: предсказание test + сохранение submit")
    test_pred = final_model.predict_proba(test_prepared)[:, 1]
    sub = pd.DataFrame({"event_id": test["event_id"].values, "predict": test_pred.astype(np.float64)})

    out_path = save_submission(
        pred_df=sub,
        sample_submit_path=sample_submit_path,
        out_dir=Path(SUBMISSIONS_DIR),
        run_name="catboost_hist_v1",
    )

    log("Готово ✅")
    print(
        {
            "valid_pr_auc": val_ap,
            "cutoff": str(cutoff),
            "saved_submit_path": out_path,
            "rows_train": int(len(train)),
            "rows_test": int(len(test)),
        }
    )


if __name__ == "__main__":
    main()
