#!/usr/bin/env bash
set -Ee -o pipefail
export PIP_USER=0
export PYTHONNOUSERSITE=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ISAACLAB_RL_FRAMEWORK="${ISAACLAB_RL_FRAMEWORK:-none}"
INSTALL_TMP=""
CURRENT_STEP=""
RUNTIME_DEPS=(
  numpy==1.26.0 packaging==23.0 typing_extensions==4.12.2 filelock==3.13.1
  websockets==12.0 scipy==1.15.3 warp-lang==1.11.0 click==8.1.7 psutil==5.9.8
  wheel==0.45.1 starlette==0.45.3 'onnx>=1.18,<1.22' 'ipython<9' virtualenv==20.30.0
)
TORCH_DEPS=(torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128)
PINK_DEPS=(
  pin==2.7.0 hpp-fcl==2.4.4 pin-pink==3.1.0 cmeel-boost==1.83.0
  cmeel-assimp==5.4.3.1 cmeel-urdfdom==3.1.1.1 eigenpy==3.5.1
  qpsolvers==4.8.2 daqp==0.7.2
)
# ── Helpers ────────────────────────────────────────────────────────────────────
info()  { echo -e "\e[1;32m>>> $*\e[0m"; }
warn()  { echo -e "\e[1;33m>>> $*\e[0m"; }
error() { echo -e "\e[1;31m[ERROR] $*\e[0m" >&2; exit 1; }
cleanup() {
  if [[ -n "$INSTALL_TMP" ]]; then
    rm -rf -- "$INSTALL_TMP"
  fi
}
install_failed() {
  local status=$?
  trap - ERR
  warn "Step '${CURRENT_STEP}' failed (exit ${status}). Fix the error, then rerun: bash scripts/install.sh --from ${CURRENT_STEP}" >&2
  exit "$status"
}
trap cleanup EXIT
trap install_failed ERR

requirements_satisfied() {
  python - "$@" <<'PY' >/dev/null 2>&1
import sys
from importlib.metadata import version, PackageNotFoundError
from packaging.requirements import Requirement

for spec in sys.argv[1:]:
    req = Requirement(spec)
    try:
        installed = version(req.name)
    except PackageNotFoundError:
        raise SystemExit(1)
    if installed not in req.specifier:
        raise SystemExit(1)
PY
}
editable_from() {
  python - "$1" "$2" <<'PY' >/dev/null 2>&1
import json
import sys
from importlib.metadata import distribution
from pathlib import Path
from urllib.parse import unquote, urlparse

data = json.loads(distribution(sys.argv[1]).read_text("direct_url.json") or "{}")
url = urlparse(data.get("url", ""))
expected = Path(sys.argv[2]).resolve()
raise SystemExit(not (
    data.get("dir_info", {}).get("editable")
    and url.scheme == "file"
    and Path(unquote(url.path)).resolve() == expected
))
PY
}
pip_install() {
  python -m pip install --constraint "$INSTALL_TMP/constraints.txt" "$@"
}
torch_stack_healthy() {
  requirements_satisfied "${TORCH_DEPS[@]}" || return 1
  python - <<'PY' >/dev/null 2>&1
import torch
import torchvision
import torchaudio
assert torch.version.cuda == "12.8"
PY
}
ensure_torch_cuda_stack() {
  if ! torch_stack_healthy; then
    warn "    Repairing PyTorch / torchvision / torchaudio cu128 stack..."
    pip_install "${TORCH_DEPS[@]}" --index-url https://download.pytorch.org/whl/cu128
    if ! torch_stack_healthy; then
      # Matching metadata can hide broken wheels. Only force reinstall if needed,
      # and keep the already resolved CUDA/runtime dependencies.
      pip_install --force-reinstall --no-deps "${TORCH_DEPS[@]}" \
        --index-url https://download.pytorch.org/whl/cu128
    fi
    torch_stack_healthy || error "PyTorch cu128 import/version check failed."
  fi
}
pin_runtime_deps() {
  if ! requirements_satisfied "${RUNTIME_DEPS[@]}"; then
    pip_install "${RUNTIME_DEPS[@]}"
  fi
  # virtualenv 21.x pulled this in; virtualenv 20.x doesn't need it and it
  # requires filelock>=3.15.4 (conflicts with isaacsim-core's filelock==3.13.1)
  if requirements_satisfied python-discovery; then
    python -m pip uninstall -y python-discovery
  fi
}
init_conda() {
  local conda_executable hook
  if [[ -x "${CONDA_EXE:-}" ]]; then
    conda_executable="$CONDA_EXE"
  elif command -v conda &>/dev/null; then
    conda_executable=conda
  elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    conda_executable="$HOME/miniconda3/bin/conda"
  else
    return 1
  fi
  hook="$("$conda_executable" shell.bash hook)" || error "Failed to initialize Conda."
  eval "$hook" || error "Failed to load Conda shell hook."
}
activate_robodojo() {
  conda activate RoboDojo || error "Cannot activate RoboDojo; run --from conda first."
  [[ "${CONDA_DEFAULT_ENV:-}" == "RoboDojo" ]] || error "Failed to activate RoboDojo."
  python - <<'PY'
import os
import sys
from pathlib import Path

if sys.version_info[:2] != (3, 11):
    raise SystemExit("RoboDojo requires Python 3.11; check the existing Conda environment.")
if Path(sys.prefix).resolve() != Path(os.environ["CONDA_PREFIX"]).resolve():
    raise SystemExit("Python does not belong to the active RoboDojo environment.")
print(f"Using Python: {sys.executable}")
PY
}
# ── Step functions ─────────────────────────────────────────────────────────────
setup_system() {
  local missing=()
  for pkg in cmake build-essential ffmpeg git git-lfs wget; do
    dpkg -s "$pkg" &>/dev/null || missing+=("$pkg")
  done
  if [ ${#missing[@]} -gt 0 ]; then
    info "[1/8] Installing system dependencies: ${missing[*]}..."
    if [ "$(id -u)" -ne 0 ]; then
      warn "    You need sudo privileges to install system packages. Please enter your password."
      sudo apt-get update && sudo apt-get install -y "${missing[@]}"
    else
      apt-get update && apt-get install -y "${missing[@]}"
    fi
  else
    warn "[1/8] System dependencies already installed, skipping..."
  fi
}
setup_conda() {
  if ! init_conda; then
    info "[2/8] Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
      -q -O "$INSTALL_TMP/miniconda.sh"
    bash "$INSTALL_TMP/miniconda.sh" -b -p "$HOME/miniconda3"
    "$HOME/miniconda3/bin/conda" init bash
    init_conda || error "Cannot initialize the new Miniconda installation."
  else
    warn "[2/8] Using existing Conda installation."
  fi
  # Miniforge and older Conda versions may not have the Anaconda ToS plugin.
  if conda tos --help &>/dev/null; then
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
  fi
  local env_list
  env_list="$(conda env list)"
  if ! grep -q '^RoboDojo[[:space:]]' <<<"$env_list"; then
    info "    Creating conda environment 'RoboDojo' (Python 3.11)..."
    conda create -n RoboDojo python=3.11 -y
  else
    warn "    Conda environment 'RoboDojo' already exists, skipping..."
  fi
  info "    Activating environment 'RoboDojo'..."
  activate_robodojo
}
setup_base_deps() {
  info "[3/8] Installing base pip dependencies..."
  pip_install -r "$CURRENT_DIR/scripts/requirements.txt" \
    opencv-python-headless==4.11.0.86 pillow matplotlib scipy==1.15.3 scikit-learn numpy==1.26.0
}
setup_submodules() {
  cd "$CURRENT_DIR" || exit 1
  local subs=(third_party/IsaacLab third_party/curobo XPolicyLab)
  info "[4/8] Initializing submodules at the commits recorded by RoboDojo..."
  # Check every initialized submodule before changing any checkout.
  for sub in "${subs[@]}"; do
    if [[ -e "$sub/.git" ]]; then
      local changes
      changes="$(git -C "$sub" status --porcelain)"
      [[ -z "$changes" ]] || error "${sub} has local changes. Preserve them before installing; no submodule was updated."
    fi
  done
  git submodule sync -- "${subs[@]}"
  git submodule update --init --progress -- "${subs[@]}"
  [ -f "XPolicyLab/client_server/ws/model_client.py" ] \
    || error "XPolicyLab init failed. Check repo access."
}
setup_isaacsim() {
  ensure_torch_cuda_stack
  if ! requirements_satisfied isaacsim==5.1.0 isaacsim-core==5.1.0 isaacsim-app==5.1.0 \
      isaacsim-kernel==5.1.0 isaacsim-extscache-physics==5.1.0 \
      isaacsim-extscache-kit==5.1.0 isaacsim-extscache-kit-sdk==5.1.0; then
    info "[5/8] Installing IsaacSim 5.1..."
    pip_install --upgrade pip
    pip_install "numpy==1.26.0" "typing_extensions==4.12.2" "filelock==3.13.1"
    pip_install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
  else
    warn "[5/8] IsaacSim 5.1.0 and extension caches already installed, skipping..."
  fi
  pin_runtime_deps
}
isaaclab_installed() {
  local source_dir
  for source_dir in "$CURRENT_DIR"/third_party/IsaacLab/source/*; do
    if [[ -f "$source_dir/setup.py" ]] && ! editable_from "${source_dir##*/}" "$source_dir"; then
      return 1
    fi
  done
}
setup_isaaclab() {
  cd "$CURRENT_DIR" || exit 1
  [[ -f third_party/IsaacLab/source/isaaclab/setup.py ]] || error "IsaacLab missing; run --from submodules first."
  if ! isaaclab_installed || [[ "$ISAACLAB_RL_FRAMEWORK" != none ]]; then
    info "[6/8] Installing IsaacLab (rl-framework: ${ISAACLAB_RL_FRAMEWORK})..."
    cd third_party/IsaacLab || error "third_party/IsaacLab not found"
    export OMNI_KIT_ACCEPT_EULA=YES
    # isaaclab.sh calls `tabs`; fails when TERM=dumb (CI / piped shells)
    export TERM=xterm-256color
    ./isaaclab.sh --install "$ISAACLAB_RL_FRAMEWORK"
    cd "$CURRENT_DIR"
    isaaclab_installed || error "Some IsaacLab extensions were not installed from this checkout."
  else
    warn "[6/8] IsaacLab extensions already installed from this checkout, skipping..."
  fi
  ensure_torch_cuda_stack
  # IsaacLab has its own dependency pins (including a different Starlette).
  # Keep its installer unchanged, then restore the Isaac Sim runtime versions.
  pin_runtime_deps
}
curobo_healthy() {
  editable_from nvidia-curobo "$CURRENT_DIR/third_party/curobo" || return 1
  python - <<'PY' &>/dev/null
from curobo.batch_motion_planner import BatchMotionPlanner, MotionPlannerCfg
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.motion_planner import MotionPlanner
from curobo.types import ToolPoseCriteria
PY
}
setup_curobo() {
  cd "$CURRENT_DIR" || exit 1
  if ! curobo_healthy; then
    info "[7/8] Installing CuRobo..."
    cd third_party/curobo || error "third_party/curobo not found"
    python -m pip uninstall -y nvidia-curobo curobo
    pip_install -e ".[cu12]" --no-build-isolation
    cd "$CURRENT_DIR"
    curobo_healthy || error "CuRobo v2 import/source check failed."
  else
    warn "[7/8] CuRobo v2 already installed from this checkout and importable, skipping..."
  fi
  pin_runtime_deps
}
pink_stack_healthy() {
  requirements_satisfied "${PINK_DEPS[@]}" numpy==1.26.0 scipy==1.15.3 || return 1
  local package
  for package in coal libcoal libpinocchio; do
    if requirements_satisfied "$package"; then
      return 1
    fi
  done
  python -c 'import pinocchio, hppfcl, pink, daqp' >/dev/null 2>&1
}
setup_pink() {
  if pink_stack_healthy; then
    warn "[8/8] Pink / Pinocchio versions and imports verified, skipping..."
    return
  fi
  info "[8/8] Repairing Pink / Pinocchio for IsaacSim 5.1 (NumPy 1.26)..."
  # Coal/libpinocchio and the older hpp-fcl/pin wheels share installed files.
  # Remove both generations before reinstalling the complete compatible stack.
  python -m pip uninstall -y coal libcoal libpinocchio pin hpp-fcl eigenpy pin-pink
  pip_install --force-reinstall "${PINK_DEPS[@]}" numpy==1.26.0 scipy==1.15.3
  python -c 'import pinocchio, hppfcl, pink, daqp'
  pink_stack_healthy || error "Pink / Pinocchio version or import check failed."
}
# ── Entry point ────────────────────────────────────────────────────────────────
usage() {
  echo "Usage: $0 [-i | --from <step>]"
  echo "  -i, --install          Full install (all steps)"
  echo "  --from <step>          Resume from a specific step:"
  echo "                           system | conda | base_deps | submodules | isaacsim | isaaclab | curobo | pink"
  echo "  ISAACLAB_RL_FRAMEWORK  IsaacLab RL extras to install (default: none; use all/sb3/skrl/etc. if needed)"
  echo "  Uses the shared RoboDojo Conda environment (Python 3.11) and recorded submodule commits."
  echo "  -h, --help             Show this help"
}
run_from() {
  local from="$1"
  local steps=(system conda base_deps submodules isaacsim isaaclab curobo pink)
  local start=0 i
  local found=0
  for i in "${!steps[@]}"; do
    if [ "${steps[$i]}" == "$from" ]; then
      start=$i
      found=1
      break
    fi
  done
  [ "$found" -eq 1 ] || error "Unknown step '$from'. Valid: ${steps[*]}"
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] \
    || error "This installer targets Linux x86_64 (Isaac Sim 5.1 / PyTorch cu128)."
  CURRENT_STEP="$from"
  INSTALL_TMP="$(mktemp -d -t robodojo-install.XXXXXXXX)"
  printf '%s\n' "${RUNTIME_DEPS[@]}" "${TORCH_DEPS[@]}" "${PINK_DEPS[@]}" \
    'stable-baselines3<2.8' >"$INSTALL_TMP/constraints.txt"
  if [ "$from" != "system" ] && [ "$from" != "conda" ]; then
    init_conda || error "Conda not found; run --from conda first."
    activate_robodojo
  fi
  for i in "${!steps[@]}"; do
    if [ "$i" -ge "$start" ]; then
      CURRENT_STEP="${steps[$i]}"
      "setup_${steps[$i]}"
    fi
  done
}
case "${1:-}" in
  -h|--help)
    [[ $# -eq 1 ]] || error "Unexpected arguments after --help."
    usage
    ;;
  -i|--install)
    [[ $# -eq 1 ]] || error "Unexpected arguments after --install."
    run_from system
    info "Develop environment setup completed."
    warn "Activate the environment: conda activate RoboDojo"
    ;;
  --from)
    [[ $# -eq 2 ]] || error "Usage: bash scripts/install.sh --from <step>"
    run_from "$2"
    info "Resumed from '$2' — done."
    warn "Activate the environment: conda activate RoboDojo"
    ;;
  *)
    usage
    exit 1
    ;;
esac

cd "$CURRENT_DIR" || exit
