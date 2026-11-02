#!/usr/bin/env bash
# Preflight checks for the selected benchmark in the shared simulator environment. Does not launch Isaac Sim or a policy server.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

policy_dir=""
dataset="MobileArena"
env_cfg=""
task_name=""
seed="0"
eval_num="native"
ckpt_name=""
action_type="ee"
sim_env="RoboDojo"
policy_env=""
skip_isaac="false"
skip_conda="false"
skip_policy="false"
summary_path=""

usage() {
  cat <<'EOF'
Usage: bash scripts/internal/verify_install.sh [options]

Options:
  --bench NAME          MobileArena (default) or SparkArena; --dataset is an alias
  --policy-dir PATH     Optional XPolicyLab policy directory to validate
  --env-cfg NAME        env_cfg stem (default: mobile_arena)
  --task NAME           Task (default: place_bread_in_oven)
  --seed NUM            Layout seed to check (default: 0)
  --eval-num NUM|native Required layout count, capped at the task default (default: native)
  --ckpt NAME|PATH      Checkpoint short name, full run name, or path (uses the runtime resolver)
  --action-type NAME    Checkpoint action type: joint or ee (default: ee, as in eval)
  --sim-env NAME        Simulator conda env for Isaac imports (default: RoboDojo)
  --policy-env NAME     Optional policy conda env to check
  --summary PATH        Write JSON summary to PATH
  --skip-isaac          Skip isaacsim/isaaclab import check
  --skip-conda          Skip conda env existence checks
  --skip-policy         Skip policy-dir/deploy/checkpoint checks
  -h, --help            Show this help
EOF
}

need_value() {
  if [[ $# -lt 2 || "$2" == --* ]]; then
    echo "[verify_install] Missing value for $1" >&2
    exit 2
  fi
}

abs_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${ROOT_DIR}/${path}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bench|--dataset) need_value "$@"; dataset="$2"; shift 2 ;;
    --policy-dir) need_value "$@"; policy_dir="$(abs_path "$2")"; shift 2 ;;
    --env-cfg) need_value "$@"; env_cfg="$2"; shift 2 ;;
    --task) need_value "$@"; task_name="$2"; shift 2 ;;
    --seed) need_value "$@"; seed="$2"; shift 2 ;;
    --eval-num) need_value "$@"; eval_num="$2"; shift 2 ;;
    --ckpt) need_value "$@"; ckpt_name="$2"; shift 2 ;;
    --action-type) need_value "$@"; action_type="$2"; shift 2 ;;
    --sim-env) need_value "$@"; sim_env="$2"; shift 2 ;;
    --policy-env) need_value "$@"; policy_env="$2"; shift 2 ;;
    --summary) need_value "$@"; summary_path="$2"; shift 2 ;;
    --skip-isaac) skip_isaac="true"; shift ;;
    --skip-conda) skip_conda="true"; shift ;;
    --skip-policy) skip_policy="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[verify_install] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "${seed}" =~ ^[0-9]+$ || ( "${eval_num}" != native && ! "${eval_num}" =~ ^[1-9][0-9]*$ ) ]]; then
  echo "[verify_install] --seed must be non-negative; --eval-num must be positive or native" >&2
  exit 2
fi

if [[ "${action_type}" != joint && "${action_type}" != ee ]]; then
  echo "[verify_install] --action-type must be joint or ee" >&2
  exit 2
fi

profile_args=()
[[ -z "${dataset}" ]] || profile_args+=(--bench "${dataset}")
[[ -z "${env_cfg}" ]] || profile_args+=(--env-cfg "${env_cfg}")
profile="$(python3 "${ROOT_DIR}/scripts/internal/benchmark_profile.py" "${profile_args[@]}")"
IFS=$'\t' read -r dataset env_cfg config_name <<< "${profile}"
if [[ -z "${task_name}" ]]; then
  task_name="$(python3 "${ROOT_DIR}/scripts/internal/benchmark_profile.py" --bench "${dataset}" --field default_task)"
fi

RESULTS_FILE="$(mktemp)"
INVENTORY_FILE="$(mktemp)"
trap 'rm -f "${RESULTS_FILE}" "${INVENTORY_FILE}"' EXIT

record() {
  local status="$1"
  local name="$2"
  local message="$3"
  printf '[%s] %s - %s\n' "${status}" "${name}" "${message}"
  printf '%s\t%s\t%s\n' "${status}" "${name}" "${message}" >> "${RESULTS_FILE}"
}

check_path() {
  local kind="$1"
  local path="$2"
  local name="$3"
  if [[ "${kind}" == "dir" && -d "${path}" ]]; then
    record "PASS" "${name}" "${path}"
  elif [[ "${kind}" == "file" && -f "${path}" ]]; then
    record "PASS" "${name}" "${path}"
  else
    record "FAIL" "${name}" "missing ${kind}: ${path}"
  fi
}

echo "[verify_install] root=${ROOT_DIR}"
echo "[verify_install] dataset=${dataset} env_cfg=${env_cfg} task=${task_name}"
if [[ -n "${policy_dir}" && "${skip_policy}" != "true" ]]; then
  echo "[verify_install] policy_dir=${policy_dir}"
else
  echo "[verify_install] policy checks disabled"
fi

check_path dir "${ROOT_DIR}/env" "env package"
check_path file "${ROOT_DIR}/scripts/evaluate.sh" "benchmark launcher"
check_path file "${ROOT_DIR}/scripts/internal/task_inventory.py" "task inventory helper"
check_path file "${ROOT_DIR}/env_cfg/${env_cfg}.yml" "env_cfg"
check_path file "${ROOT_DIR}/task/${dataset}/config/${task_name}.yml" "task config"
check_path dir "${ROOT_DIR}/Assets/Eval_Layout/${dataset}/${config_name}" "eval layouts"
if asset_info="$(python3 "${ROOT_DIR}/scripts/internal/check_assets.py" \
  --root "${ROOT_DIR}" --bench "${dataset}" --env-cfg "${env_cfg}" \
  --task "${task_name}" --seed "${seed}" --eval-num "${eval_num}" 2>&1)"; then
  record "PASS" "asset files" "${asset_info}"
else
  printf '%s\n' "${asset_info}" >&2
  record "FAIL" "asset files" "${asset_info//$'\n'/; }"
fi

if layout_info="$(python3 - "${ROOT_DIR}" "${dataset}" "${config_name}" "${task_name}" "${seed}" "${eval_num}" <<'PY'
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "scripts" / "internal"))
from benchmark_profile import expected_episode_count

bench, config_name, task, seed, requested = sys.argv[2:]
layout_dir = root / "Assets" / "Eval_Layout" / bench / config_name / seed
try:
    required = expected_episode_count(bench, task, requested)
    pattern = re.compile(rf"{re.escape(task)}_\d+\.json")
    layouts = sorted(
        (path for path in layout_dir.iterdir() if path.is_file() and pattern.fullmatch(path.name)),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    if len(layouts) < required:
        raise ValueError(f"{task}: found {len(layouts)} layouts, need {required} under {layout_dir}")
    for path in layouts[:required]:
        if not isinstance(json.loads(path.read_text(encoding="utf-8")), dict):
            raise ValueError(f"Layout must be a JSON object: {path}")
except (OSError, ValueError) as exc:
    raise SystemExit(str(exc)) from exc
print(f"{task}: {len(layouts)} layouts available, {required} required for seed {seed}")
PY
)"; then
  record "PASS" "task layouts" "${layout_info}"
else
  record "FAIL" "task layouts" "missing, incomplete, or invalid layouts for ${task_name}, seed ${seed}"
fi

if [[ -n "${policy_dir}" && "${skip_policy}" != "true" ]]; then
  check_path file "${policy_dir}/eval.sh" "policy eval.sh"
  check_path file "${policy_dir}/deploy.yml" "policy deploy.yml"
  checkpoint_args=(--policy-dir "${policy_dir}" --bench "${dataset}" --env-cfg "${env_cfg}"
    --task "${task_name}" --seed "${seed}" --action-type "${action_type}")
  [[ -z "${ckpt_name}" ]] || checkpoint_args+=(--ckpt "${ckpt_name}")
  if checkpoint_info="$(python3 "${ROOT_DIR}/scripts/internal/check_policy_checkpoint.py" "${checkpoint_args[@]}")"; then
    IFS=$'\t' read -r checkpoint_status checkpoint_message <<< "${checkpoint_info}"
    record "${checkpoint_status}" "policy checkpoint" "${checkpoint_message}"
  else
    record "FAIL" "policy checkpoint" "${checkpoint_info//$'\n'/; }"
  fi
else
  record "WARN" "policy checks" "skipped; pass --policy-dir to validate a policy"
fi

if python3 "${ROOT_DIR}/scripts/internal/task_inventory.py" --bench "${dataset}" --format json --check >"${INVENTORY_FILE}"; then
  task_counts="$(python3 - "${INVENTORY_FILE}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding='utf-8') as f:
    payload = json.load(f)
print(payload['counts'])
PY
)"
  record "PASS" "task inventory" "${task_counts}"
else
  record "FAIL" "task inventory" "task inventory check failed"
fi

if python3 - "${ROOT_DIR}" "${env_cfg}" "${dataset}" <<'PY'; then
from pathlib import Path
import sys
import yaml

root = Path(sys.argv[1])
env_cfg = sys.argv[2]
dataset = sys.argv[3]
cfg_path = root / "env_cfg" / f"{env_cfg}.yml"
cfg = yaml.safe_load(cfg_path.read_text()) or {}
if cfg.get("benchmark", "MobileArena") != dataset:
    raise SystemExit(f"env_cfg {env_cfg!r} uses benchmark {cfg.get('benchmark', 'MobileArena')!r}, not {dataset!r}")
refs = cfg.get("config", {})
required = [
    root / "env_cfg" / "sim" / f"{refs.get('sim')}.yml",
    root / "env_cfg" / "scene" / f"{refs.get('scene')}.yml",
    root / "env_cfg" / "robot" / f"{refs.get('robot')}.yml",
    root / "env_cfg" / "camera" / f"{refs.get('camera')}.yml",
    root / "env_cfg" / "robot" / "_robot_info.json",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    print("\n".join(missing), file=sys.stderr)
    raise SystemExit(1)
PY
  record "PASS" "env_cfg references" "benchmark matches; all referenced sim/scene/robot/camera files exist"
else
  record "FAIL" "env_cfg references" "invalid benchmark or missing referenced env_cfg file"
fi

if [[ -n "${policy_dir}" && "${skip_policy}" != "true" ]]; then
if python3 - "${policy_dir}" <<'PY'; then
from pathlib import Path
import sys
import yaml

policy_dir = Path(sys.argv[1])
cfg = yaml.safe_load((policy_dir / "deploy.yml").read_text()) or {}
processor = cfg.get("processor_path")
if processor:
    path = Path(processor).expanduser()
    if not path.is_absolute():
        path = policy_dir / path
    if not path.exists():
        raise SystemExit(f"missing processor_path: {path}")
PY
  record "PASS" "policy processor" "deploy.yml processor_path exists"
else
  record "FAIL" "policy processor" "deploy.yml processor_path is missing"
fi
else
  record "WARN" "policy processor" "skipped; pass --policy-dir to validate"
fi

if python3 - "${ROOT_DIR}" "${dataset}" <<'PY'; then
import sys
sys.path.insert(0, sys.argv[1])
from env.benchmark import BenchmarkPaths
assert BenchmarkPaths(sys.argv[2]).config_dir.is_dir()
PY
  record "PASS" "python import" "env.benchmark imports and resolves ${dataset}"
else
  record "FAIL" "python import" "cannot import env.benchmark or resolve ${dataset}"
fi

if [[ "${skip_conda}" == "true" ]]; then
  record "WARN" "conda envs" "skipped by --skip-conda"
elif command -v conda >/dev/null 2>&1; then
  envs="$(conda env list | awk 'NF && $1 !~ /^#/ {print $1}')"
  if grep -qx "${sim_env}" <<< "${envs}"; then
    record "PASS" "sim conda env" "${sim_env}"
  else
    record "FAIL" "sim conda env" "${sim_env} not found"
  fi
  if [[ -z "${policy_env}" ]]; then
    record "WARN" "policy conda env" "skipped; pass --policy-env to validate"
  elif [[ "${policy_env}" == "uv" || "${policy_env}" == */* ]]; then
    record "WARN" "policy conda env" "policy env is path/uv; conda check skipped"
  elif grep -qx "${policy_env}" <<< "${envs}"; then
    record "PASS" "policy conda env" "${policy_env}"
  else
    record "FAIL" "policy conda env" "${policy_env} not found"
  fi
else
  record "FAIL" "conda" "conda command not found"
fi

if [[ "${skip_isaac}" == "true" ]]; then
  record "WARN" "Isaac imports" "skipped by --skip-isaac"
elif command -v conda >/dev/null 2>&1; then
  runtime_imports="import pinocchio; import isaacsim; import isaaclab; import pink; import daqp"
  if conda run --no-capture-output -n "${sim_env}" python -c "${runtime_imports}"; then
    record "PASS" "Isaac imports" "simulator and required IK modules import in ${sim_env}"
  else
    record "FAIL" "Isaac imports" "simulator/IK import failed in ${sim_env}; see scripts/install.sh"
  fi
else
  record "FAIL" "Isaac imports" "conda unavailable; cannot check ${sim_env}"
fi

if [[ -n "${summary_path}" ]]; then
  mkdir -p "$(dirname "${summary_path}")"
  python3 - "${RESULTS_FILE}" "${summary_path}" <<'PY'
import json
from pathlib import Path
import sys

rows = []
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    status, name, message = line.split("\t", 2)
    rows.append({"status": status, "name": name, "message": message})
payload = {
    "counts": {
        "pass": sum(row["status"] == "PASS" for row in rows),
        "warn": sum(row["status"] == "WARN" for row in rows),
        "fail": sum(row["status"] == "FAIL" for row in rows),
    },
    "checks": rows,
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
  echo "[verify_install] summary=${summary_path}"
fi

fail_count="$(awk -F '\t' '$1 == "FAIL" {count++} END {print count + 0}' "${RESULTS_FILE}")"
warn_count="$(awk -F '\t' '$1 == "WARN" {count++} END {print count + 0}' "${RESULTS_FILE}")"
pass_count="$(awk -F '\t' '$1 == "PASS" {count++} END {print count + 0}' "${RESULTS_FILE}")"
echo "[verify_install] pass=${pass_count} warn=${warn_count} fail=${fail_count}"

if [[ "${fail_count}" -gt 0 ]]; then
  exit 1
fi
