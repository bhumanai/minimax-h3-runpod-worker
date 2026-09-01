#!/usr/bin/env bash
set -euo pipefail

if [[ "${MINIMAX_H3_LICENSE_ACCEPTED:-}" != "1" ]]; then
  echo "MINIMAX_H3_LICENSE_ACCEPTED=1 is required after obtaining all necessary MiniMax authorization." >&2
  exit 64
fi

python -u /download_models.py
exec /start.sh
