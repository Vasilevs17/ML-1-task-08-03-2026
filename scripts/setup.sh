#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -m pip install --upgrade pip
python -m pip install -r "$PROJECT_ROOT/requirements.txt"

"$PROJECT_ROOT/scripts/setup_data.sh"
