#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$PROJECT_ROOT/data/raw"
TMP_DIR="$PROJECT_ROOT/tmp_downloads"
ARCHIVE_PATH="$TMP_DIR/dataset_archive"

mkdir -p "$RAW_DIR" "$TMP_DIR"

if [[ -z "${GDRIVE_URL:-}" ]]; then
  echo "[ERROR] Переменная окружения GDRIVE_URL не установлена." >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "[ERROR] curl не установлен. Установите curl и повторите." >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "[ERROR] python не найден в PATH." >&2
  exit 1
fi

echo "[INFO] Скачивание архива с данными..."
curl -L --fail --silent --show-error "$GDRIVE_URL" -o "$ARCHIVE_PATH"

MIME_TYPE="$(file --brief --mime-type "$ARCHIVE_PATH")"
echo "[INFO] Определён MIME-тип: $MIME_TYPE"

extract_zip() {
  python - <<'PY' "$ARCHIVE_PATH" "$RAW_DIR"
import sys
from pathlib import Path
import zipfile

archive = Path(sys.argv[1])
out = Path(sys.argv[2])
with zipfile.ZipFile(archive) as zf:
    zf.extractall(out)
print("[INFO] ZIP архив распакован.")
PY
}

extract_tar() {
  python - <<'PY' "$ARCHIVE_PATH" "$RAW_DIR"
import sys
from pathlib import Path
import tarfile

archive = Path(sys.argv[1])
out = Path(sys.argv[2])
with tarfile.open(archive) as tf:
    tf.extractall(out)
print("[INFO] TAR архив распакован.")
PY
}

case "$MIME_TYPE" in
  application/zip)
    extract_zip
    ;;
  application/x-tar|application/gzip|application/x-gzip|application/x-bzip2|application/x-xz)
    extract_tar
    ;;
  *)
    # пробуем оба варианта на случай некорректного mime
    if python - <<'PY' "$ARCHIVE_PATH"
import sys, zipfile
try:
    zipfile.ZipFile(sys.argv[1]).testzip()
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
    then
      extract_zip
    else
      extract_tar
    fi
    ;;
esac

FILE_COUNT="$(find "$RAW_DIR" -type f | wc -l | tr -d ' ')"
if [[ "$FILE_COUNT" -eq 0 ]]; then
  echo "[ERROR] После распаковки не найдено ни одного файла в data/raw/." >&2
  exit 1
fi

echo "[INFO] Успешно: найдено файлов после распаковки: $FILE_COUNT"

# удаляем временный архив
rm -f "$ARCHIVE_PATH"
rmdir "$TMP_DIR" 2>/dev/null || true

echo "[INFO] Setup данных завершён успешно."
