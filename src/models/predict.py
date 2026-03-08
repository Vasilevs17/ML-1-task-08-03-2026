from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.data.io import load_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Инференс с сохранённой моделью")
    parser.add_argument("--model", default="models/baseline_model.joblib")
    parser.add_argument("--input", required=True, help="CSV/Parquet для предсказаний")
    parser.add_argument("--output", default="artifacts/predictions.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model)
    df = load_table(args.input)
    pred = model.predict(df)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prediction": pred}).to_csv(out, index=False)
    print(f"[INFO] Предсказания сохранены: {out}")


if __name__ == "__main__":
    main()
