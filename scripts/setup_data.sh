#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$PROJECT_ROOT"

mkdir -p downloads data/raw

EXPECTED_FILES=(
  "pretrain_part_1.parquet"
  "pretrain_part_2.parquet"
  "pretrain_part_3.parquet"
  "pretest.parquet"
  "test.parquet"
  "train_labels.parquet"
  "train_part_1.parquet"
  "train_part_2.parquet"
  "train_part_3.parquet"
)

dataset_is_complete() {
  for f in "${EXPECTED_FILES[@]}"; do
    if [ ! -f "data/raw/$f" ]; then
      return 1
    fi
  done
  return 0
}

print_existing_dataset() {
  echo "[INFO] Complete dataset already exists in data/raw, skipping download and extraction."
  for f in "${EXPECTED_FILES[@]}"; do
    if [ -f "data/raw/$f" ]; then
      echo "[INFO] data/raw/$f"
    fi
  done
}

print_missing_files() {
  echo "[INFO] Dataset is incomplete. Missing files:"
  for f in "${EXPECTED_FILES[@]}"; do
    if [ ! -f "data/raw/$f" ]; then
      echo "[INFO] missing: data/raw/$f"
    fi
  done
}

ensure_gdown() {
  if python -c 'import gdown' >/dev/null 2>&1; then
    return 0
  fi

  echo "[INFO] gdown not found, installing via pip..."
  if ! python -m pip install --quiet gdown; then
    echo "[ERROR] Failed to install gdown via pip." >&2
    exit 1
  fi

  if ! python -c 'import gdown' >/dev/null 2>&1; then
    echo "[ERROR] gdown installation looks broken: Python still cannot import gdown." >&2
    exit 1
  fi
}

# 1. Если полный набор parquet уже есть — вообще ничего не качаем
if dataset_is_complete; then
  print_existing_dataset
  exit 0
fi

print_missing_files

# 2. Если архив уже существует локально, сначала попробуем использовать его
ARCHIVE_PATH="downloads/data.zip"

archive_is_valid_zip() {
  python - <<'PY'
from pathlib import Path
import zipfile
archive = Path("downloads/data.zip")
if archive.exists() and zipfile.is_zipfile(archive):
    raise SystemExit(0)
raise SystemExit(1)
PY
}

if archive_is_valid_zip; then
  echo "[INFO] Found existing valid archive at ${ARCHIVE_PATH}, will reuse it."
else
  # 3. До этого места secret нужен только если архива нет и dataset неполный
  if [ -z "${GDRIVE_URL:-}" ]; then
    echo "[ERROR] Secret GDRIVE_URL is not set, and complete dataset is not present locally."
    exit 1
  fi

  echo "[INFO] Downloading archive with dataset..."
  ensure_gdown

  python - <<'PY'
import os
from pathlib import Path
import gdown

url = os.environ["GDRIVE_URL"]
out = Path("downloads/data.zip")
out.parent.mkdir(parents=True, exist_ok=True)

if out.exists():
    out.unlink()

try:
    result = gdown.download(url=url, output=str(out), quiet=False, fuzzy=True)
except Exception as exc:
    raise SystemExit(f"Download failed: gdown raised an exception: {exc}")

if not result:
    raise SystemExit("Download failed: gdown did not return output path")

if not out.exists():
    raise SystemExit("Download failed: downloads/data.zip was not created")

size = out.stat().st_size
if size <= 0:
    raise SystemExit("Download failed: downloads/data.zip is empty")

print(f"[INFO] Archive downloaded: {out} ({size} bytes)")
PY
fi

# 4. Распаковка
echo "[INFO] Extracting archive..."

python - <<'PY'
from pathlib import Path
import zipfile

archive = Path("downloads/data.zip")
extract_dir = Path("data/raw")
extract_dir.mkdir(parents=True, exist_ok=True)

expected_files = {
    "pretrain_part_1.parquet",
    "pretrain_part_2.parquet",
    "pretrain_part_3.parquet",
    "pretest.parquet",
    "test.parquet",
    "train_labels.parquet",
    "train_part_1.parquet",
    "train_part_2.parquet",
    "train_part_3.parquet",
}

if not archive.exists():
    raise SystemExit(f"Archive not found: {archive}")

if not zipfile.is_zipfile(archive):
    raise SystemExit(f"Invalid ZIP archive: {archive}")

with zipfile.ZipFile(archive, "r") as zf:
    zf.extractall(extract_dir)

missing_after = [f for f in sorted(expected_files) if not (extract_dir / f).exists()]
if missing_after:
    raise SystemExit(
        "Extraction finished, but dataset is still incomplete. Missing files: "
        + ", ".join(missing_after)
    )

print(f"[INFO] Dataset extracted successfully into {extract_dir}")
for f in sorted(expected_files):
    print(f"[INFO] {extract_dir / f}")
PY

# 5. Финальная проверка через bash
if dataset_is_complete; then
  echo "[INFO] Setup data completed successfully."
  echo "[INFO] Cleaning temporary archive..."
  rm -f "$ARCHIVE_PATH"
else
  echo "[ERROR] Dataset is still incomplete after extraction." >&2
  exit 1
fi
