python - <<'PY'
import os
import zipfile
from pathlib import Path

archive_path = Path("downloads/data.zip")
extract_dir = Path("data/raw")
extract_dir.mkdir(parents=True, exist_ok=True)

if not archive_path.exists():
    raise SystemExit(f"Archive not found: {archive_path}")

with zipfile.ZipFile(archive_path, "r") as zf:
    zf.extractall(extract_dir)

files = [p for p in extract_dir.rglob("*") if p.is_file()]
if not files:
    raise SystemExit("Extraction failed: no files found in data/raw")

print(f"[INFO] Extracted {len(files)} files into {extract_dir}")
PY
