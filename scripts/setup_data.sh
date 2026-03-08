#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$PROJECT_ROOT"

mkdir -p downloads data/raw

echo "[INFO] Скачивание архива с данными..."

python - <<'PY'
import os
from pathlib import Path
import gdown

url = os.environ["GDRIVE_URL"]
out = Path("downloads/data.zip")
out.parent.mkdir(parents=True, exist_ok=True)

gdown.download(url, str(out), quiet=False, fuzzy=True)

if not out.exists():
    raise SystemExit("Download failed: downloads/data.zip was not created")

if out.stat().st_size == 0:
    raise SystemExit("Download failed: downloads/data.zip is empty")

print(f"[INFO] Archive downloaded: {out} ({out.stat().st_size} bytes)")
PY

echo "[INFO] Распаковка архива..."

python - <<'PY'
from pathlib import Path
import zipfile

archive = Path("downloads/data.zip")
extract_dir = Path("data/raw")
extract_dir.mkdir(parents=True, exist_ok=True)

if not zipfile.is_zipfile(archive):
    raise SystemExit(f"Invalid ZIP archive: {archive}")

with zipfile.ZipFile(archive, "r") as zf:
    zf.extractall(extract_dir)

files = [p for p in extract_dir.rglob("*") if p.is_file()]
if not files:
    raise SystemExit("Extraction failed: no files found in data/raw")

print(f"[INFO] Extracted {len(files)} files into {extract_dir}")
PY
