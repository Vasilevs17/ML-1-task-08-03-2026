"""
ONE-CELL Colab script (v2): более сильная постановка для лидерборда.

Ключевое отличие от прошлой версии:
- train строится НЕ только из yellow/red labels,
  а из всего train-периода: red=1, yellow=0, green=0 (с контролируемым семплингом green).
Это критично для адекватного leaderboard-скора.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# =========================
# CONFIG
# =========================
DATA_DIR = "/content/drive/MyDrive/data/raw"
LABELS_ZIP = "/content/drive/MyDrive/data/train_labels.zip"
SUBMISSIONS_DIR = "/content/drive/MyDrive/data/submissions_strazh"
SAMPLE_SUBMIT_HINT = ""

RAM_LIMIT_GB = 11.5
TARGET_RAM_GB = 9.5

BATCH_SIZE_ROWS = 4_000_000
MAX_BATCH_SIZE_ROWS = 12_000_000
MIN_BATCH_SIZE_ROWS = 1_000_000

# Green (unlabeled train events) sampling to keep RAM under control
GREEN_SAMPLE_RATE = 0.06
MAX_TRAIN_ROWS = 4_800_000

VALID_SHARE = 0.2
RANDOM_STATE = 42

CATBOOST_ITERS = 2500
CATBOOST_LR = 0.04
CATBOOST_DEPTH = 8
USE_GPU = True
GPU_DEVICES = "0"

START_TS = time.time()


# =========================
# infra/logging
# =========================
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


def ram_gb() -> float:
    if psutil is None:
        return -1.0
    return float(psutil.Process().memory_info().rss / (1024**3))


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    el = _fmt_elapsed(time.time() - START_TS)
    r = ram_gb()
    rtxt = f"{r:.2f} GB" if r >= 0 else "n/a"
    print(f"[{now} | +{el} | RAM {rtxt}] {msg}")


def ram_guard(limit_gb: float = RAM_LIMIT_GB) -> None:
    r = ram_gb()
    if r >= 0 and r > limit_gb:
        raise MemoryError(f"RAM {r:.2f} GB > {limit_gb:.2f} GB. Снизьте batch/sample.")


def auto_batch(base: int = BATCH_SIZE_ROWS) -> int:
    r = ram_gb()
    if r < 0:
        return base
    if r < 3:
        x = int(base * 2.2)
    elif r < 5:
        x = int(base * 1.8)
    elif r < 7:
        x = int(base * 1.4)
    else:
        x = base
    return max(MIN_BATCH_SIZE_ROWS, min(MAX_BATCH_SIZE_ROWS, x))


# =========================
# setup helpers
# =========================
def ensure_catboost_installed() -> None:
    try:
        import catboost  # noqa

        return
    except Exception:
        log("Устанавливаю catboost...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost"])


def check_gpu_runtime() -> None:
    """Быстрая проверка наличия GPU в runtime Colab."""
    if not USE_GPU:
        log("GPU режим отключён (USE_GPU=False)")
        return

    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.STDOUT, text=True)
        log("GPU detected: " + out.strip().splitlines()[0])
    except Exception as e:
        raise RuntimeError(
            "GPU не обнаружена. В Colab включите Runtime -> Change runtime type -> GPU."
        ) from e


def smoke_test_catboost_gpu() -> None:
    """Мини-тест CatBoost GPU до тяжёлой подготовки данных."""
    if not USE_GPU:
        return
    ensure_catboost_installed()
    from catboost import CatBoostClassifier

    X = pd.DataFrame({
        "num": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "cat": ["a", "b", "a", "b", "a", "b"],
    })
    y = pd.Series([0, 1, 0, 1, 0, 1], dtype="int8")

    try:
        m = CatBoostClassifier(
            iterations=5,
            depth=4,
            learning_rate=0.2,
            loss_function="Logloss",
            verbose=False,
            task_type="GPU",
            devices=GPU_DEVICES,
            allow_writing_files=False,
        )
        m.fit(X, y, cat_features=["cat"])
        _ = m.predict_proba(X)
        log("CatBoost GPU smoke-test: OK")
    except Exception as e:
        raise RuntimeError(
            "GPU обнаружена, но CatBoost GPU fit не стартует. "
            "Проверьте runtime Colab (GPU) и перезапустите kernel."
        ) from e


def mount_drive() -> None:
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive", force_remount=False)


def resolve_sample_submit_path(preferred: str = "") -> str:
    cand = []
    if preferred:
        cand.append(Path(preferred))
    cand += [
        Path("/content/drive/MyDrive/data/sample_submit.csv"),
        Path("/content/drive/MyDrive/sample_submit.csv"),
        Path("/content/ML-1-task-08-03-2026/sample_submit.csv"),
    ]
    for p in cand:
        if p.exists():
            return str(p)
    raise FileNotFoundError("Не найден sample_submit.csv: " + ", ".join(map(str, cand)))


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
    miss = sorted(expected - existing)
    if not miss:
        return
    if not labels_zip.exists():
        raise FileNotFoundError(f"Не хватает parquet ({miss}) и zip не найден: {labels_zip}")
    import zipfile

    log(f"Распаковка zip, отсутствует {len(miss)} parquet...")
    with zipfile.ZipFile(labels_zip, "r") as zf:
        zf.extractall(data_dir)


# =========================
# features
# =========================
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


def prepare_for_catboost(df: pd.DataFrame, feat_cols: List[str], cat_cols: List[str]) -> pd.DataFrame:
    out = df[feat_cols].copy().replace({None: np.nan})
    obj_like = [c for c in out.columns if str(out[c].dtype) in ("object", "string", "category")]
    all_cat = sorted(set(cat_cols).union(obj_like))

    for c in all_cat:
        out[c] = out[c].astype("string").fillna("__MISSING__").astype(str)
        out[c] = out[c].replace({"<NA>": "__MISSING__", "nan": "__MISSING__", "None": "__MISSING__"})

    num_cols = [c for c in out.columns if c not in all_cat]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
        med = out[c].median()
        out[c] = out[c].fillna(0.0 if pd.isna(med) else float(med))

    return out


# =========================
# dataset build
# =========================
def load_labels(data_dir: Path) -> pd.DataFrame:
    labels = pd.read_parquet(data_dir / "train_labels.parquet", columns=["customer_id", "event_id", "target"])
    labels["customer_id"] = labels["customer_id"].astype("int64")
    labels["event_id"] = labels["event_id"].astype("int64")
    labels["target"] = labels["target"].astype("int8")
    return labels


def build_train_all_with_green(
    data_dir: Path,
    labels: pd.DataFrame,
    batch_size: int,
    green_sample_rate: float,
    max_rows: int,
) -> pd.DataFrame:
    """Собирает train: red=1, yellow=0, green=0 (green с семплингом)."""
    rng = np.random.default_rng(RANDOM_STATE)
    parts = []
    total_kept = 0

    labels_small = labels.copy()
    for p in sorted(data_dir.glob("train_part_*.parquet")):
        log(f"Сбор train из {p.name} ...")
        pf = pq.ParquetFile(p)
        processed = 0
        total = pf.metadata.num_rows

        for rb in pf.iter_batches(batch_size=batch_size):
            b = rb.to_pandas()
            m = b.merge(labels_small, on=["customer_id", "event_id"], how="left", indicator=True)

            matched = m["_merge"].eq("both")
            m["target"] = m["target"].fillna(0).astype("int8")

            # keep all labeled; sample unlabeled green
            green = ~matched
            keep_green = rng.random(len(m)) < green_sample_rate
            keep_mask = matched | (green & keep_green)
            k = m.loc[keep_mask].drop(columns=["_merge"])

            parts.append(k)
            total_kept += len(k)
            processed += len(b)

            if processed % (batch_size * 2) == 0 or processed == total:
                log(f"{p.name}: {processed:,}/{total:,}, kept={total_kept:,}")
                ram_guard()

            if total_kept >= max_rows:
                log(f"Достигнут лимит train rows={max_rows:,}, останавливаю дальнейший сбор")
                break

            del b, m, k
            gc.collect()

        if total_kept >= max_rows:
            break

    if not parts:
        raise RuntimeError("Пустой train после сборки")

    train = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()

    # дедуп по ключу, если пересечения (на всякий)
    train = train.drop_duplicates(["customer_id", "event_id"], keep="first").copy()
    log(f"train all(with green) = {train.shape}")
    return train


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

    # Более широкие customer stats для усиления качества
    cnt: Dict[int, float] = {}
    amt_sum: Dict[int, float] = {}
    amt_sumsq: Dict[int, float] = {}
    amt_min: Dict[int, float] = {}
    amt_max: Dict[int, float] = {}
    hour_sum: Dict[int, float] = {}
    weekend_sum: Dict[int, float] = {}
    evt_nuniq: Dict[int, set] = {}

    def upd_sum(d: Dict[int, float], s: pd.Series):
        for k, v in s.items():
            d[int(k)] = d.get(int(k), 0.0) + float(v)

    for fp in files:
        if not fp.exists():
            continue
        log(f"History scan: {fp.name}")
        pf = pq.ParquetFile(fp)
        total = pf.metadata.num_rows
        processed = 0

        for rb in pf.iter_batches(batch_size=batch_size, columns=["customer_id", "operaton_amt", "event_dttm", "event_type_nm"]):
            b = rb.to_pandas()
            b["event_dttm"] = pd.to_datetime(b["event_dttm"], errors="coerce")
            b["event_hour"] = b["event_dttm"].dt.hour.astype("float32")
            b["is_weekend"] = (b["event_dttm"].dt.dayofweek >= 5).astype("float32")
            b["operaton_amt"] = pd.to_numeric(b["operaton_amt"], errors="coerce")

            g_cnt = b.groupby("customer_id").size().astype("float64")
            upd_sum(cnt, g_cnt)

            amt = b.dropna(subset=["operaton_amt"])
            if not amt.empty:
                upd_sum(amt_sum, amt.groupby("customer_id")["operaton_amt"].sum())
                upd_sum(amt_sumsq, amt.assign(_sq=amt["operaton_amt"] ** 2).groupby("customer_id")["_sq"].sum())
                for k, v in amt.groupby("customer_id")["operaton_amt"].min().items():
                    kk, vv = int(k), float(v)
                    if kk not in amt_min or vv < amt_min[kk]:
                        amt_min[kk] = vv
                for k, v in amt.groupby("customer_id")["operaton_amt"].max().items():
                    kk, vv = int(k), float(v)
                    if kk not in amt_max or vv > amt_max[kk]:
                        amt_max[kk] = vv

            upd_sum(hour_sum, b.dropna(subset=["event_hour"]).groupby("customer_id")["event_hour"].sum())
            upd_sum(weekend_sum, b.dropna(subset=["is_weekend"]).groupby("customer_id")["is_weekend"].sum())

            # event_type uniq (ограниченно)
            e = b[["customer_id", "event_type_nm"]].dropna()
            for cid, grp in e.groupby("customer_id"):
                c = int(cid)
                if c not in evt_nuniq:
                    evt_nuniq[c] = set()
                if len(evt_nuniq[c]) < 256:
                    evt_nuniq[c].update(grp["event_type_nm"].astype(int).tolist())

            processed += len(b)
            if processed % (batch_size * 2) == 0 or processed == total:
                log(f"{fp.name}: {processed:,}/{total:,}")
                ram_guard()

            del b
            gc.collect()

    cids = sorted(cnt.keys())
    hist = pd.DataFrame({"customer_id": cids})
    hist["cust_txn_cnt"] = hist["customer_id"].map(lambda x: cnt.get(int(x), 0.0)).astype("float32")
    hist["cust_amt_sum"] = hist["customer_id"].map(lambda x: amt_sum.get(int(x), 0.0)).astype("float32")
    hist["cust_amt_min"] = hist["customer_id"].map(lambda x: amt_min.get(int(x), np.nan)).astype("float32")
    hist["cust_amt_max"] = hist["customer_id"].map(lambda x: amt_max.get(int(x), np.nan)).astype("float32")
    hist["cust_hour_sum"] = hist["customer_id"].map(lambda x: hour_sum.get(int(x), 0.0)).astype("float32")
    hist["cust_weekend_sum"] = hist["customer_id"].map(lambda x: weekend_sum.get(int(x), 0.0)).astype("float32")
    hist["cust_evt_nuniq"] = hist["customer_id"].map(lambda x: float(len(evt_nuniq.get(int(x), set())))).astype("float32")

    denom = hist["cust_txn_cnt"].replace(0, np.nan)
    hist["cust_amt_mean"] = (hist["cust_amt_sum"] / denom).astype("float32")
    hist["cust_hour_mean"] = (hist["cust_hour_sum"] / denom).astype("float32")
    hist["cust_weekend_rate"] = (hist["cust_weekend_sum"] / denom).astype("float32")

    # std
    hist["cust_amt_sumsq"] = hist["customer_id"].map(lambda x: amt_sumsq.get(int(x), 0.0)).astype("float32")
    var = (hist["cust_amt_sumsq"] / denom) - (hist["cust_amt_mean"] ** 2)
    hist["cust_amt_std"] = np.sqrt(np.clip(var, 0, None)).astype("float32")
    hist.drop(columns=["cust_amt_sumsq"], inplace=True)

    log(f"history shape={hist.shape}")
    return hist


# =========================
# train/predict
# =========================
def time_aware_split(df: pd.DataFrame, valid_share: float = VALID_SHARE):
    w = df.copy()
    w["event_dttm"] = pd.to_datetime(w["event_dttm"], errors="coerce")
    cutoff = w["event_dttm"].quantile(1.0 - valid_share)
    tr = w[w["event_dttm"] < cutoff].copy()
    va = w[w["event_dttm"] >= cutoff].copy()
    return tr.drop(columns=["target"]), tr["target"].astype("int8"), va.drop(columns=["target"]), va["target"].astype("int8"), cutoff


def fit_catboost(Xtr: pd.DataFrame, ytr: pd.Series, Xva: pd.DataFrame, yva: pd.Series):
    ensure_catboost_installed()
    from catboost import CatBoostClassifier

    feat_cols = [c for c in Xtr.columns if c not in ["event_id", "event_dttm"]]
    cat_cols = [c for c in feat_cols if str(Xtr[c].dtype) in ("object", "string", "category")]

    Xtrp = prepare_for_catboost(Xtr, feat_cols, cat_cols)
    Xvap = prepare_for_catboost(Xva, feat_cols, cat_cols)

    pos_w = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
    log(f"CatBoost fit: Xtr={Xtrp.shape}, Xva={Xvap.shape}, cat={len(cat_cols)}, spw={pos_w:.3f}")

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        iterations=CATBOOST_ITERS,
        learning_rate=CATBOOST_LR,
        depth=CATBOOST_DEPTH,
        l2_leaf_reg=7.0,
        random_strength=1.5,
        subsample=0.8,
        bootstrap_type="Bernoulli",
        scale_pos_weight=pos_w,
        random_seed=RANDOM_STATE,
        verbose=200,
        od_type="Iter",
        od_wait=300,
        allow_writing_files=False,
        task_type=("GPU" if USE_GPU else "CPU"),
        devices=GPU_DEVICES if USE_GPU else None,
    )
    model.fit(Xtrp, ytr, eval_set=(Xvap, yva), cat_features=cat_cols, use_best_model=True)

    from sklearn.metrics import average_precision_score

    va_pred = model.predict_proba(Xvap)[:, 1]
    ap = float(average_precision_score(yva, va_pred))
    return model, feat_cols, cat_cols, ap


def save_submission(sub: pd.DataFrame, sample_submit: str, out_dir: Path, run_name: str) -> str:
    smp = pd.read_csv(sample_submit)
    if set(["event_id", "predict"]) - set(smp.columns):
        raise ValueError("sample_submit должен содержать event_id,predict")
    if len(sub) != len(smp):
        raise ValueError(f"len submit {len(sub)} != {len(smp)}")
    if set(sub["event_id"].astype(str)) != set(smp["event_id"].astype(str)):
        raise ValueError("event_id submit != sample_submit")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"submit__catboost_green_v2__{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(out, index=False)
    return str(out)


def main() -> None:
    log("Шаг 0: mount drive")
    mount_drive()

    check_gpu_runtime()
    smoke_test_catboost_gpu()

    data_dir = Path(DATA_DIR)
    ensure_data_present(data_dir, Path(LABELS_ZIP))
    sample_submit = resolve_sample_submit_path(SAMPLE_SUBMIT_HINT)
    log(f"sample_submit={sample_submit}")

    labels = load_labels(data_dir)
    log(f"labels={labels.shape}, target_mean={labels['target'].mean():.5f}")

    b1 = auto_batch(BATCH_SIZE_ROWS)
    log(f"Шаг 1: сбор train c green-sampling (CPU I/O этап), batch={b1:,}, green_rate={GREEN_SAMPLE_RATE}")
    train = build_train_all_with_green(
        data_dir,
        labels,
        batch_size=b1,
        green_sample_rate=GREEN_SAMPLE_RATE,
        max_rows=MAX_TRAIN_ROWS,
    )
    del labels
    gc.collect()
    ram_guard()

    b2 = auto_batch(max(BATCH_SIZE_ROWS, b1))
    log(f"Шаг 2: history features (CPU I/O этап), batch={b2:,}")
    hist = build_customer_history_features(data_dir, batch_size=b2)

    log("Шаг 3: merge + row features")
    train = add_row_features(train)
    train = train.merge(hist, on="customer_id", how="left")

    test = pd.read_parquet(data_dir / "test.parquet")
    test = add_row_features(test)
    test = test.merge(hist, on="customer_id", how="left")

    del hist
    gc.collect()
    ram_guard()

    log("Шаг 4: time-aware split")
    Xtr, ytr, Xva, yva, cutoff = time_aware_split(train, valid_share=VALID_SHARE)
    log(f"split cutoff={cutoff}; train={len(Xtr):,}; valid={len(Xva):,}")

    log("Шаг 5: train CatBoost (GPU)")
    model, feat_cols, cat_cols, valid_ap = fit_catboost(Xtr, ytr, Xva, yva)
    log(f"VALID PR-AUC={valid_ap:.6f}")

    log("Шаг 6: final fit on full train")
    from catboost import CatBoostClassifier

    full_X = prepare_for_catboost(train, feat_cols, cat_cols)
    test_X = prepare_for_catboost(test, feat_cols, cat_cols)
    full_y = train["target"].astype("int8")

    best_it = int(model.get_best_iteration()) if model.get_best_iteration() is not None else 700
    pos_w = float((full_y == 0).sum()) / max(float((full_y == 1).sum()), 1.0)
    final_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        iterations=max(1200, int((best_it + 1) * 1.30)),
        learning_rate=CATBOOST_LR,
        depth=CATBOOST_DEPTH,
        l2_leaf_reg=7.0,
        random_strength=1.5,
        subsample=0.8,
        bootstrap_type="Bernoulli",
        scale_pos_weight=pos_w,
        random_seed=RANDOM_STATE,
        verbose=200,
        allow_writing_files=False,
        task_type=("GPU" if USE_GPU else "CPU"),
        devices=GPU_DEVICES if USE_GPU else None,
    )
    final_model.fit(full_X, full_y, cat_features=cat_cols)

    log("Шаг 7: predict + save submit")
    pred = final_model.predict_proba(test_X)[:, 1]
    sub = pd.DataFrame({"event_id": test["event_id"].values, "predict": pred.astype(np.float64)})

    out_path = save_submission(sub, sample_submit, Path(SUBMISSIONS_DIR), "catboost_green_v2")
    log("Готово ✅")
    print(
        {
            "valid_pr_auc": valid_ap,
            "rows_train": int(len(train)),
            "rows_test": int(len(test)),
            "saved_submit_path": out_path,
        }
    )


if __name__ == "__main__":
    main()
