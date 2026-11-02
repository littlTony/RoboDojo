#!/usr/bin/env bash
# Shared benchmark CLI; benchmark metadata selects tasks, profiles, and summaries.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash scripts/evaluate.sh <command> [--bench NAME] [options]
       bash scripts/evaluate.sh --bench NAME <command> [options]

Benchmarks: MobileArena (default), SparkArena. --dataset is an alias for --bench.
Default profiles: mobile_arena, tianji_marvin_wuji, respectively.
The simulator Conda environment is RoboDojo.

Commands:
  doctor      Check benchmark assets/configs/environment
  eval        Evaluate one task with a fresh policy server
  server      Start a policy server; omit --task for batch server mode
  client      Connect to external servers; omit --task for a task sweep
  smoke       Evaluate selected/all tasks (default: one episode each)
  benchmark   Evaluate selected/all tasks with --eval-num NUM|native
  tasks       Inspect canonical runnable tasks
  summarize   Aggregate results using the benchmark's existing scoring rules

Batch commands share task filtering and multi-GPU scheduling. Run a command
with --help for options. MobileArena/mobile_arena.sh remains a compatibility alias.
EOF
}

abs_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${ROOT_DIR}/${path}"
  fi
}

need_value() {
  if [[ $# -lt 2 || "$2" == --* ]]; then
    echo "[bench] Missing value for $1" >&2
    exit 2
  fi
}

policy_eval_uses_expert_num() {
  local eval_script="$1"
  grep -Eq 'expert_data_num|expert_num' "${eval_script}"
}

resolve_policy_name() {
  local policy_dir="$1"
  basename "${policy_dir}"
}

validate_policy_dir() {
  local policy_dir="$1"
  local label="${2:-policy}"
  if [[ ! -f "${policy_dir}/setup_eval_policy_server.sh" ]]; then
    echo "[bench ${label}] setup_eval_policy_server.sh not found: ${policy_dir}/setup_eval_policy_server.sh" >&2
    exit 1
  fi
}

# probe_tcp HOST PORT TIMEOUT_SECONDS -> exit 0 if a TCP connect succeeds.
# Uses bash /dev/tcp so it needs no extra tools; `timeout` bounds the wait.
probe_tcp() {
  local host="$1"
  local port="$2"
  local timeout_s="${3:-5}"
  timeout "${timeout_s}" bash -c ">/dev/tcp/${host}/${port}" 2>/dev/null
}

run_doctor() {
  bash "${ROOT_DIR}/scripts/internal/verify_install.sh" --bench "${BENCH_NAME}" "$@"
}

run_tasks() {
  python3 "${ROOT_DIR}/scripts/internal/task_inventory.py" --bench "${BENCH_NAME}" "$@"
}

run_eval() {
  local dataset="${BENCH_NAME}"
  local task=""
  local ckpt=""
  local env_cfg="${BENCH_ENV_CFG}"
  local expert_num="100"
  local action_type="ee"
  local seed="0"
  local policy_gpu="0"
  local env_gpu="0"
  local policy_env=""
  local eval_env="RoboDojo"
  local policy_dir=""
  local eval_num="${EVAL_NUM:-}"
  local dry_run="false"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --task) need_value "$@"; task="$2"; shift 2 ;;
      --ckpt) need_value "$@"; ckpt="$2"; shift 2 ;;
      --env-cfg) need_value "$@"; env_cfg="$2"; shift 2 ;;
      --expert-num) need_value "$@"; expert_num="$2"; shift 2 ;;
      --action-type) need_value "$@"; action_type="$2"; shift 2 ;;
      --seed) need_value "$@"; seed="$2"; shift 2 ;;
      --policy-gpu) need_value "$@"; policy_gpu="$2"; shift 2 ;;
      --env-gpu) need_value "$@"; env_gpu="$2"; shift 2 ;;
      --policy-env) need_value "$@"; policy_env="$2"; shift 2 ;;
      --eval-env) need_value "$@"; eval_env="$2"; shift 2 ;;
      --policy-dir) need_value "$@"; policy_dir="$(abs_path "$2")"; shift 2 ;;
      --eval-num) need_value "$@"; eval_num="$2"; shift 2 ;;
      --dry-run) dry_run="true"; shift ;;
      -h|--help)
        cat <<'EOF'
Usage: bash scripts/evaluate.sh eval --policy-dir PATH --task TASK --ckpt CKPT --policy-env ENV [options]

Required:
  --policy-dir PATH     XPolicyLab policy directory containing eval.sh
  --task TASK           Selected benchmark task name
  --ckpt CKPT           Policy checkpoint name
  --policy-env ENV      Policy conda env, uv, or env path

Common options:
  --bench NAME          MobileArena (default) or SparkArena; --dataset is an alias
  --eval-num NUM|native  Override EVAL_NUM for this eval; use `native` for per-task counts from _task.yml
  --env-cfg NAME        env_cfg stem (default: mobile_arena)
  --expert-num NUM      Legacy policy argument (default: 100)
  --action-type NAME    Policy action type (default: ee)
  --seed NUM            Eval seed / layout seed (default: 0)
  --policy-gpu ID       Policy server GPU (default: 0)
  --env-gpu ID          Isaac Sim GPU (default: 0)
  --eval-env ENV        Simulator conda env (default: RoboDojo)
  --dry-run             Print command without running it

Split / multi-machine: use `evaluate.sh server` + `evaluate.sh client` with the same --bench.
EOF
        return 0
        ;;
      *)
        echo "[bench eval] Unknown argument: $1" >&2
        exit 2
        ;;
    esac
  done

  if [[ -z "${policy_dir}" || -z "${task}" || -z "${ckpt}" || -z "${policy_env}" ]]; then
    echo "[bench eval] --policy-dir, --task, --ckpt, and --policy-env are required" >&2
    exit 2
  fi
  if [[ ! -f "${policy_dir}/eval.sh" ]]; then
    echo "[bench eval] policy eval.sh not found: ${policy_dir}/eval.sh" >&2
    exit 1
  fi

  local eval_args=()
  if policy_eval_uses_expert_num "${policy_dir}/eval.sh"; then
    eval_args=(
      "${dataset}"
      "${task}"
      "${ckpt}"
      "${env_cfg}"
      "${expert_num}"
      "${action_type}"
      "${seed}"
      "${policy_gpu}"
      "${env_gpu}"
      "${policy_env}"
      "${eval_env}"
    )
  else
    eval_args=(
      "${dataset}"
      "${task}"
      "${ckpt}"
      "${env_cfg}"
      "${action_type}"
      "${seed}"
      "${policy_gpu}"
      "${env_gpu}"
      "${policy_env}"
      "${eval_env}"
    )
  fi

  apply_eval_num "${eval_num}"

  echo "[bench eval] policy_dir=${policy_dir}"
  echo "[bench eval] task=${task} env_cfg=${env_cfg} eval_num=${EVAL_NUM:-default}"

  if [[ "${dry_run}" == "true" ]]; then
    printf '[bench eval] dry-run: bash %q' "${ROOT_DIR}/scripts/internal/run_policy_eval.sh"
    printf ' %q' "${policy_dir}"
    printf ' %q' "${eval_args[@]}"
    printf '\n'
    return 0
  fi

  exec bash "${ROOT_DIR}/scripts/internal/run_policy_eval.sh" "${policy_dir}" "${eval_args[@]}"
}

run_server() {
  local dataset="${BENCH_NAME}"
  local task=""
  local ckpt=""
  local env_cfg="${BENCH_ENV_CFG}"
  local action_type="ee"
  local seed="0"
  local policy_gpu="0"
  local gpu_ids=""
  local policy_gpu_ids=""
  local policy_env=""
  local policy_dir=""
  local policy_port=""
  local bind_host="0.0.0.0"
  local dry_run="false"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --task) need_value "$@"; task="$2"; shift 2 ;;
      --ckpt) need_value "$@"; ckpt="$2"; shift 2 ;;
      --env-cfg) need_value "$@"; env_cfg="$2"; shift 2 ;;
      --action-type) need_value "$@"; action_type="$2"; shift 2 ;;
      --seed) need_value "$@"; seed="$2"; shift 2 ;;
      --policy-gpu) need_value "$@"; policy_gpu="$2"; shift 2 ;;
      --gpu-ids) need_value "$@"; gpu_ids="$2"; shift 2 ;;
      --policy-gpu-ids) need_value "$@"; policy_gpu_ids="$2"; shift 2 ;;
      --policy-env) need_value "$@"; policy_env="$2"; shift 2 ;;
      --policy-dir) need_value "$@"; policy_dir="$(abs_path "$2")"; shift 2 ;;
      --policy-port) need_value "$@"; policy_port="$2"; shift 2 ;;
      --bind-host) need_value "$@"; bind_host="$2"; shift 2 ;;
      --dry-run) dry_run="true"; shift ;;
      -h|--help)
        cat <<'EOF'
Usage: bash scripts/evaluate.sh server --bench NAME --policy-dir PATH --ckpt CKPT --policy-env ENV [options]

Starts only the XPolicyLab policy WebSocket server (foreground). Use with
`evaluate.sh client` on another host or container with the same --bench.

Required:
  --policy-dir PATH     XPolicyLab policy directory (XPolicyLab/policy/<NAME>)
  --ckpt CKPT           Policy checkpoint name
  --policy-env ENV      Policy conda env, uv, or env path

Common options:
  --bench NAME          MobileArena (default) or SparkArena; --dataset is an alias
  --task TASK           Task passed to the policy server (omit for batch mode)
  --policy-port PORT    TCP port (auto-selected in single-task mode; required in batch)
  --bind-host HOST      Bind address (default: 0.0.0.0 for remote clients; use localhost for local-only)
  --env-cfg NAME        env_cfg stem (default: mobile_arena)
  --action-type NAME    Policy action type (default: ee)
  --seed NUM            Eval seed (default: 0)
  --policy-gpu ID       Policy server GPU (default: 0)
  --gpu-ids IDS         Worker GPU ids in batch mode; one id in single-task mode
  --policy-gpu-ids IDS   Worker policy GPUs; one id in single-task mode
  --dry-run             Print command without running it

Batch mode: omit --task and use --only, --tasks-file, --limit,
--gpu-ids, and --policy-port PORTS. See `evaluate.sh benchmark --help` for all options.
EOF
        return 0
        ;;
      *)
        echo "[bench server] Unknown argument: $1" >&2
        exit 2
        ;;
    esac
  done

  if [[ -n "${gpu_ids}" ]]; then
    if [[ "${gpu_ids}" == *","* ]]; then
      echo "[bench server] multiple --gpu-ids are not supported" >&2
      exit 2
    fi
    policy_gpu="${gpu_ids}"
  fi
  if [[ -n "${policy_gpu_ids}" ]]; then
    if [[ "${policy_gpu_ids}" == *","* ]]; then
      echo "[bench server] multiple --policy-gpu-ids are not supported" >&2
      exit 2
    fi
    policy_gpu="${policy_gpu_ids}"
  fi

  if [[ -z "${policy_dir}" || -z "${task}" || -z "${ckpt}" || -z "${policy_env}" ]]; then
    echo "[bench server] --policy-dir, --task, --ckpt, and --policy-env are required" >&2
    exit 2
  fi
  validate_policy_dir "${policy_dir}" "server"
  if [[ "${policy_port}" == *","* || "${bind_host}" == *","* ]]; then
    echo "[bench server] comma-separated --policy-port/--bind-host are not supported" >&2
    exit 2
  fi

  local server_args=(
    "${policy_dir}"
    "${dataset}"
    "${task}"
    "${ckpt}"
    "${env_cfg}"
    "${action_type}"
    "${seed}"
    "${policy_gpu}"
    "${policy_env}"
    "${policy_port}"
    "${bind_host}"
  )

  echo "[bench server] policy_dir=${policy_dir} task=${task} bind_host=${bind_host} port=${policy_port:-auto}"

  if [[ "${dry_run}" == "true" ]]; then
    printf '[bench server] dry-run: bash %q' "${ROOT_DIR}/scripts/internal/run_policy_server.sh"
    printf ' %q' "${server_args[@]}"
    printf '\n'
    return 0
  fi

  exec bash "${ROOT_DIR}/scripts/internal/run_policy_server.sh" "${server_args[@]}"
}

run_client() {
  local dataset="${BENCH_NAME}"
  local task=""
  local env_cfg="${BENCH_ENV_CFG}"
  local policy_name=""
  local policy_dir=""
  local policy_host="127.0.0.1"
  local policy_port=""
  local seed="0"
  local env_gpu="0"
  local gpu_ids=""
  local env_gpu_ids=""
  local ckpt="external"
  local action_type="ee"
  local eval_num="${EVAL_NUM:-}"
  local eval_batch=""
  local connect_timeout="5"
  local dry_run="false"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --task) need_value "$@"; task="$2"; shift 2 ;;
      --env-cfg) need_value "$@"; env_cfg="$2"; shift 2 ;;
      --policy-name) need_value "$@"; policy_name="$2"; shift 2 ;;
      --policy-dir) need_value "$@"; policy_dir="$(abs_path "$2")"; shift 2 ;;
      --policy-host) need_value "$@"; policy_host="$2"; shift 2 ;;
      --policy-port) need_value "$@"; policy_port="$2"; shift 2 ;;
      --seed) need_value "$@"; seed="$2"; shift 2 ;;
      --env-gpu) need_value "$@"; env_gpu="$2"; shift 2 ;;
      --gpu-ids) need_value "$@"; gpu_ids="$2"; shift 2 ;;
      --env-gpu-ids) need_value "$@"; env_gpu_ids="$2"; shift 2 ;;
      --ckpt) need_value "$@"; ckpt="$2"; shift 2 ;;
      --action-type) need_value "$@"; action_type="$2"; shift 2 ;;
      --eval-num) need_value "$@"; eval_num="$2"; shift 2 ;;
      --eval-batch) need_value "$@"; eval_batch="$2"; shift 2 ;;
      --connect-timeout) need_value "$@"; connect_timeout="$2"; shift 2 ;;
      --dry-run) dry_run="true"; shift ;;
      -h|--help)
        cat <<'EOF'
Usage: bash scripts/evaluate.sh client --bench NAME (--policy-name NAME | --policy-dir PATH) --policy-port PORT [options]

Runs the selected benchmark simulator client against an already-running external policy
server. Pair with `evaluate.sh server` on the policy machine with the same --bench.

Batch mode:
  Omit `--task` and pass task filters such as `--only`, `--tasks-file`, or
  `--limit`, together with `--gpu-ids` or `--env-gpu-ids`. Tasks will be
  grouped using the benchmark scheduler and launched across client workers.
  If `--policy-host` contains multiple hosts, its count and `--policy-port`
  count must both equal the group count. If `--policy-host` contains one host,
  `--policy-port` count must equal the group count and the host is reused.

Required:
  --policy-name NAME     Policy name recorded in results; model code stays on the server
                         Omit when --policy-dir is set (name inferred from directory)
  --policy-dir PATH      Same as eval's --policy-dir; sets --policy-name from basename
  --policy-port PORT     Policy server TCP port

Common options:
  --bench NAME          MobileArena (default) or SparkArena; --dataset is an alias
  --task TASK           Selected benchmark task name (omit for batch mode)
  --policy-host HOST    Policy server host (default: 127.0.0.1)
  --eval-num NUM|native  Override EVAL_NUM for this run; `native` uses per-task counts
  --eval-batch BOOL      External server accepts batches: true or false (default: local deploy.yml)
  --env-cfg NAME         env_cfg stem (default: mobile_arena)
  --seed NUM             Eval seed / layout seed (default: 0)
  --env-gpu ID           Isaac Sim GPU (default: 0)
  --gpu-ids IDS          Worker GPU ids in batch mode; one id in single-task mode
  --env-gpu-ids IDS      Worker client GPUs; one id in single-task mode
  --ckpt NAME            Checkpoint label recorded in result paths (default: external)
  --action-type NAME     Action type label recorded in result paths (default: ee)
  --connect-timeout SEC  Pre-flight policy-server reachability probe timeout (default: 5)
  --only a,b,c           Batch mode task subset
  --tasks-file PATH      Batch mode task subset file
  --limit NUM            Batch mode task limit after filtering
  --dry-run              Print the resolved eval_policy.sh command without running it
EOF
        return 0
        ;;
      *)
        echo "[bench client] Unknown argument: $1" >&2
        exit 2
        ;;
    esac
  done

  if [[ -n "${policy_dir}" ]]; then
    validate_policy_dir "${policy_dir}" "client"
    if [[ -z "${policy_name}" ]]; then
      policy_name="$(resolve_policy_name "${policy_dir}")"
    fi
  fi

  if [[ -z "${policy_name}" ]]; then
    echo "[bench client] --policy-name or --policy-dir is required" >&2
    exit 2
  fi

  if [[ -n "${eval_batch}" && "${eval_batch}" != true && "${eval_batch}" != false ]]; then
    echo "[bench client] --eval-batch must be true or false" >&2
    exit 2
  fi

  if [[ -z "${policy_host}" || -z "${policy_port}" ]]; then
    echo "[bench client] single-task mode requires --task, --policy-host, and --policy-port" >&2
    exit 2
  fi
  if [[ "${policy_host}" == *","* || "${policy_port}" == *","* ]]; then
    echo "[bench client] comma-separated --policy-host/--policy-port are only supported in batch mode" >&2
    exit 2
  fi

  if [[ -n "${gpu_ids}" ]]; then
    env_gpu="${gpu_ids}"
  fi
  if [[ -n "${env_gpu_ids}" ]]; then
    env_gpu="${env_gpu_ids}"
  fi
  if [[ "${env_gpu}" == *","* ]]; then
    echo "[bench client] multiple GPU ids require batch mode (omit --task)" >&2
    exit 2
  fi

  local additional_info="ckpt_name=${ckpt},action_type=${action_type}"

  apply_eval_num "${eval_num}"

  local client_args=(
    --dataset_name "${dataset}"
    --task_name "${task}"
    --env_cfg_type "${env_cfg}"
    --policy_name "${policy_name}"
    --host "${policy_host}"
    --port "${policy_port}"
    --protocol ws
    --root_dir "${ROOT_DIR}"
    --device_id "${env_gpu}"
    --additional_info "${additional_info}"
    --seed "${seed}"
  )
  if [[ -n "${eval_batch}" ]]; then
    client_args+=(--eval_batch "${eval_batch}")
  fi

  echo "[bench client] task=${task} policy=${policy_name} server=${policy_host}:${policy_port} eval_num=${EVAL_NUM:-default}"

  if [[ "${dry_run}" == "true" ]]; then
    printf '[bench client] dry-run: bash %q' "${ROOT_DIR}/scripts/eval_policy.sh"
    printf ' %q' "${client_args[@]}"
    printf '\n'
    return 0
  fi

  if probe_tcp "${policy_host}" "${policy_port}" "${connect_timeout}"; then
    echo "[bench client] policy server reachable at ${policy_host}:${policy_port}"
  else
    echo "[bench client] WARNING: could not reach ${policy_host}:${policy_port} within ${connect_timeout}s" >&2
    echo "[bench client] The client will keep retrying; verify that:" >&2
    echo "[bench client]   - the policy server is running and bound to 0.0.0.0 (not its own localhost)" >&2
    echo "[bench client]   - in Docker use --network host, or --policy-host host.docker.internal on a bridge network" >&2
    echo "[bench client]   - the port is correct and not blocked by a firewall" >&2
  fi

  exec bash "${ROOT_DIR}/scripts/eval_policy.sh" "${client_args[@]}"
}

run_sweep() {
  local mode="$1"
  shift
  for arg in "$@"; do
    if [[ "${arg}" == "-h" || "${arg}" == "--help" ]]; then
      bash "${ROOT_DIR}/scripts/internal/smoke_all_tasks.sh" --bench "${BENCH_NAME}" "$@"
      return
    fi
  done
  if [[ "${mode}" == "smoke" ]]; then
    exec bash "${ROOT_DIR}/scripts/internal/smoke_all_tasks.sh" --bench "${BENCH_NAME}" --eval-num "${EVAL_NUM:-1}" "$@"
  else
    local has_eval_num="false"
    for arg in "$@"; do
      if [[ "${arg}" == "--eval-num" ]]; then
        has_eval_num="true"
        break
      fi
    done
    if [[ "${has_eval_num}" != "true" && -z "${EVAL_NUM:-}" ]]; then
      echo "[bench benchmark] pass --eval-num NUM|native or set EVAL_NUM" >&2
      exit 2
    fi
    exec bash "${ROOT_DIR}/scripts/internal/smoke_all_tasks.sh" --bench "${BENCH_NAME}" "$@"
  fi
}

apply_eval_num() {
  local value="$1"
  if [[ "${value}" == native ]]; then
    unset EVAL_NUM
  elif [[ -n "${value}" ]]; then
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
      echo "[bench] --eval-num / EVAL_NUM must be a positive integer or native" >&2
      exit 2
    fi
    export EVAL_NUM="${value}"
  fi
}

run_single_or_batch() {
  local env_cfg="" task="" resolved result_config
  local args=()
  for arg in "$@"; do
    if [[ "${arg}" == -h || "${arg}" == --help ]]; then
      "run_${command}" --help
      return 0
    fi
  done
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --env-cfg) need_value "$@"; env_cfg="$2"; shift 2 ;;
      --task) need_value "$@"; task="$2"; args+=("$1" "$2"); shift 2 ;;
      *) args+=("$1"); shift ;;
    esac
  done
  resolved="$(python3 "${ROOT_DIR}/scripts/internal/benchmark_profile.py" \
    --bench "${BENCH_NAME}" --env-cfg "${env_cfg}")"
  IFS=$'\t' read -r BENCH_NAME BENCH_ENV_CFG result_config <<< "${resolved}"

  if [[ -z "${task}" && "${command}" != eval ]]; then
    local batch_args=(--mode "${command}" --bench "${BENCH_NAME}" --env-cfg "${BENCH_ENV_CFG}")
    if [[ "${command}" == client ]]; then
      batch_args+=(--policy-host 127.0.0.1)
    fi
    exec bash "${ROOT_DIR}/scripts/internal/smoke_all_tasks.sh" "${batch_args[@]}" ${args[@]+"${args[@]}"}
  fi
  if [[ -n "${task}" ]]; then
    local task_names
    task_names="$(python3 "${ROOT_DIR}/scripts/internal/task_inventory.py" --bench "${BENCH_NAME}" --only-runnable)"
    if ! grep -Fxq -- "${task}" <<< "${task_names}"; then
      echo "[bench ${command}] Unknown ${BENCH_NAME} task: ${task}" >&2
      exit 2
    fi
  fi
  "run_${command}" ${args[@]+"${args[@]}"}
}

run_summarize() {
  local summary_script
  summary_script="$(python3 "${ROOT_DIR}/scripts/internal/benchmark_profile.py" --bench "${BENCH_NAME}" --field summary_script)"
  python3 "${ROOT_DIR}/${summary_script}" "$@"
}

BENCH_NAME="MobileArena"
BENCH_ENV_CFG=""
command=""
command_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bench|--dataset) need_value "$@"; BENCH_NAME="$2"; shift 2 ;;
    --bench=*|--dataset=*) BENCH_NAME="${1#*=}"; shift ;;
    *)
      if [[ -z "${command}" ]]; then
        command="$1"
      else
        command_args+=("$1")
      fi
      shift
      ;;
  esac
done
set -- ${command_args[@]+"${command_args[@]}"}

case "${command}" in
  doctor) run_doctor "$@" ;;
  tasks) run_tasks "$@" ;;
  eval|server|client) run_single_or_batch "$@" ;;
  smoke) run_sweep smoke "$@" ;;
  benchmark) run_sweep benchmark "$@" ;;
  summarize) run_summarize "$@" ;;
  -h|--help) usage ;;
  "") usage; exit 2 ;;
  *)
    echo "[bench] Unknown command: ${command}" >&2
    usage >&2
    exit 2
    ;;
esac
