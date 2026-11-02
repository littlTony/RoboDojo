#!/usr/bin/env bash
# Download MobileArena assets or import an existing SparkArena asset package.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bench="MobileArena"
source_path=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bench|--dataset|--source)
      if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        echo "[assets] Missing value for $1" >&2
        exit 2
      fi
      if [[ "$1" == --source ]]; then source_path="$2"; else bench="$2"; fi
      shift 2
      ;;
    -h|--help)
      echo "Usage: bash scripts/init_assets.sh [--bench MobileArena|SparkArena] [--source /path/to/Assets]"
      echo "MobileArena uses the shared Git/LFS source. SparkArena requires an existing Assets source directory."
      exit 0
      ;;
    *) echo "[assets] Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "${bench}" == SparkArena ]]; then
  exec bash "${SCRIPT_DIR}/SparkArena/import_assets.sh" --source "${source_path}"
fi
if [[ -n "${source_path}" ]]; then
  echo "[assets] --source supports SparkArena only" >&2
  exit 2
fi
exec python3 "${SCRIPT_DIR}/internal/assets.py" --bench "${bench}"
