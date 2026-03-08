#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$PROJECT_ROOT"

if [ -z "${GDRIVE_URL:-}" ]; then
  echo "[ERROR] Secret GDRIVE_URL is not set"
  exit 1
fi

mkdir -p downloads data/raw

# Если данные уже распакованы, повторно не качаем
if find data/raw -type f | grep -q .; then
  echo "[INFO] Files already exist in data/raw, skipping download and extraction."
  find data/raw -maxdepth 2 -type f | head -20
  exit 0
fi

echo "[INFO] Скачивание архива с данными..."

python - <<'PY'
import os
from pathlib import Path
import gdown

url = os.environ["GDRIVE_URL"]
out = Path("downloads/data.zip")
out.parent.mkdir(parents=True, exist_ok=True)

if out.exists():
    out.unlink()

gdown.download(url, str(out), quiet=False, fuzzy=True)

if not out.exists():
    raise SystemExit("Download failed: downloads/data.zip was not created")

size = out.stat().st_size
if size <= 0:
    raise SystemExit("Download failed: downloads/data.zip is empty")

print(f"[INFO] Archive downloaded: {out} ({size} bytes)")
PY

echo "[INFO] Распаковка архива..."

python - <<'PY'
from pathlib import Path
import zipfile

archive = Path("downloads/data.zip")
extract_dir = Path("data/raw")
extract_dir.mkdir(parents=True, exist_ok=True)

if not archive.exists():
    raise SystemExit(f"Archive not found: {archive}")

if not zipfile.is_zipfile(archive):
    raise SystemExit(f"Invalid ZIP archive: {archive}")

with zipfile.ZipFile(archive, "r") as zf:
    zf.extractall(extract_dir)

files = [p for p in extract_dir.rglob("*") if p.is_file()]
if not files:
    raise SystemExit("Extraction failed: no files found in data/raw")

print(f"[INFO] Extracted {len(files)} files into {extract_dir}")
for p in files[:20]:
    print(f"[INFO] {p}")
PY

echo "[INFO] Очистка временного архива..."
rm -f downloads/data.zip

echo "[INFO] Setup data completed successfully."
