# MobileArena evaluation

This checkout evaluates **MobileArena**: eight mobile manipulation tasks and
their `_l` (language) and `_s` (spatial) variants, for 24 runnable tasks.
It uses NVIDIA Isaac Sim / IsaacLab and XPolicyLab policy servers.

MobileArena is the default benchmark for
`bash scripts/evaluate.sh <command>`. Explicit `--bench MobileArena` is optional.
See the [task and policy guide](scripts/MobileArena/README.md),
[CLI reference](scripts/README.md), and [Docker guide](docker/README.md).

**SparkArena** adds seven tabletop tasks for `tianji_marvin_wuji`. Select
`--bench SparkArena` to connect the simulator to an existing policy server;
see the [SparkArena guide](scripts/SparkArena/README.md) for asset import,
single-task and batch evaluation. Model code and weights stay on the server.

The shared simulation framework derives from RoboDojo. The repository directory,
simulator Conda environment (`RoboDojo`), shared runtime environment variables,
and external asset path placeholders retain their existing names.
Local assets and previous evaluation results are preserved.

Policy integration lives in [XPolicyLab](XPolicyLab/README.md).

## MobileArena quick start

The simulator installer targets **Linux x86_64**, an NVIDIA GPU, and Python 3.11.
Run these commands from the repository root. Policy dependencies are installed
separately from the simulator environment.

```bash
git clone https://github.com/ZixuanLi2004/RoboDojo.git
cd RoboDojo
git submodule update --init --recursive
bash scripts/install.sh --install
conda activate RoboDojo
bash scripts/init_assets.sh --bench MobileArena
```

`init_assets.sh` downloads MobileArena objects, rooms, robots, and saved layouts
together with shared assets. Its source is configured through `HF_REPO_ID`,
`HF_REPO_URL`, and `HF_REVISION`. The selected revision must contain MobileArena
assets; missing packages cause an error. Assets are not bundled with this checkout.
See the [asset options](scripts/README.md#download-assets) and the
[MobileArena asset layout](scripts/MobileArena/README.md#installation-and-assets).

Prepare the included weight-free demo policy, then check one task and preview
its launch:

```bash
conda create -n demo-policy python=3.11 -y
conda activate demo-policy
bash XPolicyLab/policy/demo_policy/install.sh
conda activate RoboDojo

bash scripts/evaluate.sh tasks --bench MobileArena --check
bash scripts/evaluate.sh doctor --bench MobileArena \
  --task place_kettle_on_tv_stand --seed 0 --eval-num 1 \
  --policy-dir XPolicyLab/policy/demo_policy --policy-env demo-policy
bash scripts/evaluate.sh eval --bench MobileArena \
  --policy-dir XPolicyLab/policy/demo_policy \
  --task place_kettle_on_tv_stand --ckpt demo --policy-env demo-policy \
  --action-type joint --seed 0 --eval-num 1 --dry-run
```

After the required assets and environments are ready, remove `--dry-run` for
an actual one-episode interface check. `demo_policy` holds the observed posture
and commands zero base velocity; it does not solve the task or load a checkpoint.
For learned policies, install the selected adapter's dependencies and supply a
compatible checkpoint. The [policy compatibility list](scripts/MobileArena/README.md#policy-and-action-compatibility)
specifies which adapters require `joint` or `ee`; the CLI default is `ee`.

Selecting `--bench MobileArena` sets the default environment profile to
`mobile_arena` (`env_cfg/mobile_arena.yml`), with `config_name: mobile_arena` for layouts and
results. Results are written under
`eval_result/MobileArena/<task>/<policy>/mobile_arena/`. Use the
[MobileArena guide](scripts/MobileArena/README.md) for multi-GPU evaluation, split
server/client commands, resume, and the full 24-task × 3-seed summary.
All 24 tasks, including the base tasks and `_l`/`_s` variants, require 25 episodes
per seed, as configured in `task/MobileArena/config/_task.yml`.

### Checks without simulation

With Python 3.11+, PyYAML, NumPy, and Bash 4+ available:

```bash
bash -n scripts/evaluate.sh scripts/eval_policy.sh
bash scripts/evaluate.sh doctor --skip-isaac --skip-conda --skip-policy
bash scripts/evaluate.sh tasks --bench MobileArena --check
```

These checks validate task/configuration discovery and available asset files.
They do not start Isaac Sim or load policy weights. Use an actual one-episode
evaluation to verify asset loading, physics, rendering, and the policy connection.

## 🗂️ Repository Structure

```text
env/                   simulator backbone and managers
env_cfg/               simulator, scene, robot, and camera configs
task/MobileArena/       24 mobile manipulation tasks and configs (8 base + 16 variants)
src/eval_client/       shared simulator client
scripts/evaluate.sh    MobileArena CLI (default benchmark)
scripts/internal/      shared task discovery, validation, launch, and batching
scripts/MobileArena/    MobileArena scoring, guide, and compatibility CLI alias
scripts/eval_policy.sh shared simulator client entry
XPolicyLab/            policy server and policy integrations
Assets/                downloaded robot, object, room, material, and layout assets
```

`evaluate.sh summarize` writes `eval_result/MobileArena/_summary.md` by default.
Full scoring covers 24 tasks × three seeds, with 25 episodes per task and seed.
Invalid or incomplete latest results remain incomplete; older results are not
used as a fallback.

## 🔌 Policy Integration

Policies live in the [XPolicyLab submodule](XPolicyLab/README.md), which owns policy structure, dependencies, checkpoint layout, and server behavior. The shared evaluation path uses these policy-side files:

```text
XPolicyLab/policy/<POLICY_NAME>/eval.sh
XPolicyLab/policy/<POLICY_NAME>/setup_eval_policy_server.sh
XPolicyLab/policy/<POLICY_NAME>/setup_eval_env_client.sh
XPolicyLab/policy/<POLICY_NAME>/deploy.py
XPolicyLab/policy/<POLICY_NAME>/deploy.yml
```

`evaluate.sh eval` requires `eval.sh` for validation but launches the two setup
scripts directly. MobileArena uses profile `mobile_arena` and result paths under
`eval_result/MobileArena/`. The client setup
calls `scripts/eval_policy.sh`; `deploy.py`
provides the environment-side adapter, and `deploy.yml` declares transport,
batching, and policy-specific runtime settings. A policy's `eval.sh` is also
available as a direct policy-side entry point.

## Acknowledgements

This evaluation stack builds on RoboDojo, Isaac Sim, IsaacLab, CuRobo, and
XPolicyLab. Their original licenses and copyright notices are retained.

## ⚖️ License

Released under the [MIT License](LICENSE).


<p align="center">
  <img alt="Python 3.11" src="https://img.shields.io/badge/Python-3.11-475569?style=flat-square&logo=python&logoColor=white&labelColor=64748b" height="22"/>&nbsp;
  <img alt="Isaac Sim 5.1" src="https://img.shields.io/badge/Isaac_Sim-5.1-475569?style=flat-square&logo=nvidia&logoColor=76B900&labelColor=64748b" height="22"/>&nbsp;
  <img alt="Isaac Lab 2.3" src="https://img.shields.io/badge/Isaac_Lab-2.3-475569?style=flat-square&logo=nvidia&logoColor=76B900&labelColor=64748b" height="22"/>&nbsp;
  <img alt="License MIT" src="https://img.shields.io/badge/License-MIT-475569?style=flat-square&labelColor=64748b" height="22"/>
</p>
