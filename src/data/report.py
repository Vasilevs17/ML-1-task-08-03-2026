from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.io import SUPPORTED_EXTENSIONS, load_table


def guess_target_candidates(df: pd.DataFrame) -> list[str]:
    candidates = []
    for col in df.columns:
        name = col.lower()
        if any(token in name for token in ["target", "label", "y", "class", "price", "predict"]):
            candidates.append(col)
    if not candidates and len(df.columns) > 0:
        candidates.append(df.columns[-1])
    return candidates[:5]


def summarize_file(path: Path) -> dict:
    df = load_table(path)
    mem_mb = float(df.memory_usage(deep=True).sum() / (1024**2))
    missing_ratio = float(df.isna().sum().sum() / (df.shape[0] * max(df.shape[1], 1)))
    return {
        "file": str(path),
        "format": path.suffix.lower(),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "memory_mb": round(mem_mb, 3),
        "target_candidates": guess_target_candidates(df),
        "missing_ratio": round(missing_ratio, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Быстрый отчёт по данным в data/raw")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output", default="artifacts/data_report.json")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    files = sorted(
        p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise SystemExit("CSV/Parquet файлы не найдены в data/raw")

    summary = [summarize_file(p) for p in files]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Отчёт сохранён: {out}")


if __name__ == "__main__":
    main()
