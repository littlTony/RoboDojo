#!/usr/bin/env bash
# Import the SparkArena dependency set without overwriting existing assets.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_path=""
if [[ $# -eq 2 && "$1" == --source && -n "$2" ]]; then
  source_path="$2"
fi
if [[ ! -d "${source_path}" ]]; then
  echo "[assets] SparkArena requires --source /path/to/Assets" >&2
  exit 2
fi
source_path="$(cd "${source_path}" && pwd -P)"
target_path="${ROOT_DIR}/Assets"
if [[ "${source_path}" == "$(realpath -m "${target_path}")" ]]; then
  echo "[assets] Source and target Assets must differ" >&2
  exit 2
fi
required=(
  Robots/tianji_marvin_wuji
  Object/SparkArena
  Eval_Layout/SparkArena/tianji_marvin_wuji
  Room/Simple_Room_nolight
  Material/material_0116
  Material/material_0122
  Background/brown_photostudio_02_4k.hdr
)
for path in "${required[@]}"; do
  if [[ ! -e "${source_path}/${path}" ]]; then
    echo "[assets] Missing source asset: ${source_path}/${path}" >&2
    exit 1
  fi
done
for path in "${required[@]}"; do
  mkdir -p "${target_path}/$(dirname "${path}")"
  cp -aLn --reflink=auto -- "${source_path}/${path}" "${target_path}/$(dirname "${path}")/"
done
cd "${ROOT_DIR}"
python3 utils/update_embodiment_config_path.py
echo "[assets] SparkArena assets imported into ${target_path}"
