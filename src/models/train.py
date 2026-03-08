from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import yaml

from src.data.io import load_first_supported_file, load_table
from src.data.validation import validate_frame
from src.features.preprocessing import split_features_target
from src.models.pipeline import train_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение baseline-модели")
    parser.add_argument("--config", default="configs/train.yaml", help="Путь к YAML-конфигу")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["data"]["raw_dir"])
    data_file = cfg["data"].get("file")
    target_col = cfg["data"]["target_col"]

    if data_file:
        source_path = raw_dir / data_file
        df = load_table(source_path)
    else:
        source_path, df = load_first_supported_file(raw_dir)

    report = validate_frame(df)

    split = split_features_target(df, target_col)
    result = train_baseline(
        split.X,
        split.y,
        test_size=float(cfg["train"].get("test_size", 0.2)),
        random_state=int(cfg["train"].get("random_state", 42)),
    )

    model_dir = Path(cfg["outputs"]["models_dir"])
    metrics_dir = Path(cfg["outputs"]["artifacts_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "baseline_model.joblib"
    metrics_path = metrics_dir / "metrics.json"

    joblib.dump(result.model, model_path)

    payload = {
        "source_file": str(source_path),
        "validation": report.__dict__,
        "metrics": result.metrics,
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] Модель сохранена: {model_path}")
    print(f"[INFO] Метрики сохранены: {metrics_path}")


if __name__ == "__main__":
    main()
