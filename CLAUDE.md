# MobileArena — Claude Code Guide

## Project Overview

This checkout evaluates MobileArena and SparkArena on the shared Isaac Sim / IsaacLab stack.
MobileArena remains the default. Use `scripts/evaluate.sh` for inventory, validation, launches, and
batch evaluation. The repository name and `RoboDojo` simulator Conda environment
remain unchanged. Local assets and old evaluation results are retained.

Contributors maintain **Tasks** (scene + reward logic), shared **env** / **env_cfg** configs, and eval infrastructure under **scripts/** and **src/eval_client/**.

Human-facing docs: [MobileArena](scripts/MobileArena/README.md), [SparkArena](scripts/SparkArena/README.md).
SparkArena uses an existing external policy server; retain the current submodule
pins and keep policy model code and weights on that server.

## Repository Layout

```
task/MobileArena/                added mobile manipulation task logic and configs
task/task_registry.py        shared benchmark-scoped dynamic task loader
scripts/MobileArena/             MobileArena scoring, guide, compatibility CLI alias
env/                         TaskEnv backbone and managers
env_cfg/                     robot / scene / sim / camera YAML
src/eval_client/             eval client (main.py, eval_env.py)
utils/                       paths, save/load, pipeline helpers
scripts/                     evaluate.sh, install.sh, init_assets.sh, internal/
docker/                      container eval (see docker/README.md)
XPolicyLab/                  policy server + deploy (submodule)
third_party/                 IsaacLab, curobo (submodules)
Assets/                      downloaded runtime assets for the selected benchmarks (not in git)
eval_result/                 runtime eval output
```

## Submodules & Ownership

| Own in RoboDojo | Own in XPolicyLab |
| :-- | :-- |
| `env/`, `env_cfg/`, `task/MobileArena/`, `src/eval_client/` | Policy code, checkpoints, `deploy.yml`, policy server |
| `scripts/evaluate.sh`, `scripts/eval_policy.sh` | `XPolicyLab/policy/<POLICY>/eval.sh`, `setup_eval_*` scripts |
| `utils/`, install/assets scripts | Policy-specific dependencies and training |

- Submodule URL: `https://github.com/ZixuanLi2004/XPolicyLab.git` (branch `main`).
- Update the gitlink pin intentionally; do not edit submodule contents unless updating the pin.
- RoboDojo eval client uses `XPolicyLab/client_server/ws/model_client.py` (WebSocket transport).

## Eval Commands & Flow

**CLI** (`bash scripts/evaluate.sh <command> --bench MobileArena`):

MobileArena is the default, so `--bench MobileArena` is optional. `--dataset`
is an alias for `--bench`. `scripts/MobileArena/mobile_arena.sh` forwards to the
same CLI. Defaults and summary dispatch live in `scripts/internal/benchmark_profile.py`;
round-robin GPU grouping lives in `scripts/internal/benchmark_groups.py`.

| Command | Purpose |
| :-- | :-- |
| `doctor` | Check assets, configs, env before eval |
| `eval` | One task: policy server + sim client on localhost |
| `server` | Policy server only (split / multi-machine eval) |
| `client` | Sim client only (connect to running server) |
| `smoke` | Selected/all tasks, `EVAL_NUM=1` by default |
| `benchmark` | Full sweep (`--eval-num native` or numeric); sequential unless GPU lists are supplied |
| `summarize` | Aggregate `eval_result/` into markdown table |
| `tasks` | List runnable tasks (maintainer inventory) |

**Single-machine eval:**

```
evaluate.sh eval
  → scripts/internal/run_policy_eval.sh
    → (cd policy_dir) setup_eval_policy_server.sh   # policy server
    → setup_eval_env_client.sh                    # sim client → src/eval_client/main.py
```

- Policy `eval.sh` must exist under `--policy-dir` (used for validation and optional `expert_num`); `evaluate.sh eval` does **not** invoke `eval.sh` directly.
- Policy `setup_eval_*` scripts run with **CWD = policy directory** (relative paths in policies).
- Default transport: **WebSocket** (`protocol: ws` in policy `deploy.yml`; see `scripts/eval_policy.sh`).
- Explicitly choose the action contract. MobileArena FastWAM/Pi_05/MM_ABC/MoPA use `joint`; OpenWAM/Xiaomi_Robotics_1 use `ee`; demo supports either. Checkpoint layouts are adapter-specific even when their vectors have the same width.
- Policy installation is separate from simulator installation. The demo environment can be prepared with `conda create -n demo-policy python=3.11`, activation, then `bash XPolicyLab/policy/demo_policy/install.sh`.

**Split eval:**

```
evaluate.sh server  →  scripts/internal/run_policy_server.sh
evaluate.sh client  →  scripts/eval_policy.sh  →  src/eval_client/main.py
```

**Docker:** sim client in container, policy server on host — see `docker/README.md`.

Fast validation:

```bash
bash scripts/evaluate.sh doctor --skip-isaac --skip-conda --skip-policy
bash scripts/evaluate.sh eval --policy-dir XPolicyLab/policy/<POLICY> --task place_bread_in_oven --ckpt <CKPT> --policy-env <ENV> --dry-run
bash scripts/evaluate.sh smoke --policy-dir XPolicyLab/policy/<POLICY> --ckpt <CKPT> --policy-env <ENV> --only place_bread_in_oven,throw_away_trash --dry-run
```

For a sequential eval-infrastructure acceptance run:

```bash
bash scripts/evaluate.sh smoke --policy-dir XPolicyLab/policy/<POLICY> --ckpt <CKPT> --policy-env <ENV> --fail-fast
```

Multi-GPU batching uses round-robin task assignment; equal task counts do not
guarantee equal execution time. Scoring lives in `scripts/MobileArena/summarize_result.py`
and writes `eval_result/MobileArena/_summary.md` by default.

New code imports from `env.*`.

## Tasks

MobileArena adds **24 tasks** under `task/MobileArena/`: eight base tasks and their `_l`
(language) and `_s` (spatial) variants. Use
`bash scripts/evaluate.sh tasks --bench MobileArena --check` to inspect them. Its default
profile is `mobile_arena`, with layout/result `config_name: mobile_arena`.
Registration loads `task.MobileArena.tasks.<task_name>` and expects the exported
class name to match the module. Configs live in `task/MobileArena/config/`.

## Naming

- **Config / scripts**: lowercase `snake_case` — `env_cfg/`, `utils/`, `scripts/`
- **Python packages/directories**: `snake_case` — `env/robot_manager/`, `env/scene_manager/objects/`
- **Python module files**: `snake_case.py` — `task_env.py`, `obs_manager.py`
- **Python classes**: **PascalCase** — `RewardManager`, `TaskEnv`
- **Tasks**: filename, YAML name, exported env class, and layout/result paths must match. Use lowercase `snake_case` for task names.
- **Submodules**: keep upstream casing — `XPolicyLab/`, `Assets/`

## Code Conventions

### Task File Structure

Every task Python file must follow this pattern:

```python
class <TaskName>Common:
    def __init__(self, config, app, **kwargs): ...
    def _post_setup_scene(self, sim): ...
    def reset(self, seed=None, options=None): ...
    def soft_reset(self, seed=None, options=None): ...  # when state must reset between episodes
    def run_reward(self): ...                           # reward / success check

class <task_name>(<TaskName>Common, TaskEnv):
    pass
```

- Class names use **PascalCase**; the exported env class uses **snake_case** matching the YAML / Python module basename.
- `run_reward()` must call `self.reward_manager.check(...)` (or equivalent) meaningfully — never leave success checks trivially always `True`.
- No print statements in task logic; use the shared logger if needed.

### Config File (YAML)

Typical rigid-body tasks define:
- `Rigid` section with at least one object category and a `label` list.
- `common` with `xlim`, `ylim`, `rotate_rand` (and `rotate_deg` when rotation is enabled).
- `select_mode` with `nums` and `mode`.

Task configs may also declare `Articulation`, `Geometry`, and other object sections.

### Commit Message Format

```
[Scope] type: short description

Scopes: Task | Env | Config | scripts | utils | fix | refactor
Types : feat | fix | update | refactor | docs | chore
```

Examples:
- `[Task] update: fix place_bread_in_oven reward check`
- `[Env] fix: correct gripper warm-up timing`
- `[Config] update: adjust place_fruit_basket configuration`

### Branch Naming

`<contributor>/<scope>-<short-description>`  e.g. `zixuan/task-place-bread`

## Validation Loops

Use the **smallest loop** that proves the current change.

### Docs / scripts / config only

```bash
bash -n scripts/evaluate.sh scripts/eval_policy.sh
python scripts/internal/task_inventory.py --bench MobileArena --format json --check
bash scripts/evaluate.sh doctor --skip-isaac --skip-conda --skip-policy
ruff check .
git diff --check
```

CPU tests require Python 3.11+, PyYAML, NumPy, and Bash 4+. They validate CLI
contracts and selected action-layout/seed/result behavior without launching
Isaac Sim or loading checkpoints. Do not describe dry runs or these tests as
proof of GPU simulation, asset loading, Pink IK, or actual model inference.

Asset checks require real assets. `scripts/init_assets.sh` selects MobileArena
and shared packages by default. The Hugging Face asset source is configurable through `HF_REPO_ID`, `HF_REPO_URL`, and `HF_REVISION`.
The selected revision must contain the requested benchmark's assets; missing
packages must fail rather than appear ready. MobileArena needs complete
`Assets/Object/MobileArena/`, `Assets/Room/MobileArena/`, `Assets/Robots/m92uw/`, and
`Assets/Eval_Layout/MobileArena/mobile_arena/<seed>/` packages, plus shared materials and
backgrounds. Download verification checks file lists, sizes, and unresolved
Git LFS pointers; it does not prove GPU simulation or native layout coverage.
Check each requested task/seed/count with `evaluate.sh doctor --bench BENCH
--task TASK --seed N --eval-num NUM|native`; omit skip flags for a full preflight.
Doctor asset checks follow task-specific scene/robot overrides and saved layout
references, rejecting missing/empty files and unresolved LFS pointers. Keep
these checks offline; USD dependency loading still requires a real smoke test.
Doctor checkpoint checks use the XPolicyLab runtime resolver and the same
`--action-type` (default `ee`) and seed as evaluation. Accept short names,
full run names and paths; checking path existence is not weight validation.

### Dry-run eval path (no Isaac, no policy server)

```bash
bash scripts/evaluate.sh eval \
  --policy-dir XPolicyLab/policy/<POLICY_NAME> \
  --task place_bread_in_oven \
  --ckpt <CKPT_NAME> \
  --policy-env <POLICY_ENV> \
  --dry-run
bash scripts/evaluate.sh smoke \
  --policy-dir XPolicyLab/policy/<POLICY_NAME> \
  --ckpt <CKPT_NAME> \
  --policy-env <POLICY_ENV> \
  --only place_bread_in_oven,throw_away_trash \
  --dry-run
```

### Runtime (Isaac Sim + policy ready)

```bash
bash scripts/evaluate.sh eval \
  --policy-dir XPolicyLab/policy/<POLICY_NAME> \
  --task place_bread_in_oven \
  --ckpt <CKPT_NAME> \
  --policy-env <POLICY_ENV> \
  --eval-num 1
```

Smoke/benchmark acceptance: each task must exit `0` **and** write `_result.json`
with a valid integer `eval_time >= min(requested, native)`. `native` uses the
task's declared count in `_task.yml`: 25 for all MobileArena base tasks and their
`_l`/`_s` variants. Partial runs must fail completion checks and must not be skipped
on resume. This is run completion, not policy task success.
The simulator's single-task exit status follows the same completion requirement;
preserve its resume manifest on incomplete runs, and remove it only on completion.

Resume identities include benchmark, profile, policy, checkpoint, action type,
seed, episode count, and execution mode (`eval` or `client`). Persistent server
mode does not support resume. In split deployment, match these policy settings
and the port mapping across server/client; standalone clients run in the active
simulator environment and only label results with `--ckpt` / `--action-type`.

MobileArena defaults to one environment per simulator; `EVAL_NUM_ENVS` overrides
that count when the adapter has `eval_batch: true`. `eval_batch: false` forces
one environment. Multi-GPU groups are separate processes and must not be confused
with parallel environments within a process.

MobileArena summaries use 24 tasks × seeds `0,1,2` (72 cells), requiring 25 valid
episode records per base-task and `_l`/`_s` variant cell.
`summarize --env-cfg` filters result `config_name`; launch `--env-cfg` selects
the profile stem. Both values are `mobile_arena` in the default profile. Select the
most recently modified `_result.json` per experiment/task/seed, breaking ties by
run-directory name. Invalid or incomplete latest results must not silently fall
back to older complete runs.

## PR Checklist

### Task PR
- [ ] `config/<task>.yml` + `tasks/<task>.py` both present, names match
- [ ] `run_reward()` calls `self.reward_manager.check(...)` meaningfully
- [ ] YAML labels match Python labels exactly
- [ ] `step_lim` is appropriate for task length
- [ ] Task inherits `TaskEnv` (not `SyncCollectEnv` / `SyncRobotEnv`)
- [ ] New task filenames are lowercase `snake_case`

### Framework / Env PR
- [ ] Backward-compatible; no new mandatory constructor args
- [ ] `reset()` and `soft_reset()` both cover new state when applicable

### Config PR
- [ ] YAML valid; defaults safe; no silent breakage to shared configs

### General
- [ ] Commit messages: `[Scope] type: desc`
- [ ] No `print`, `breakpoint`, or commented-out debug code
- [ ] New files follow Naming rules above
- [ ] For scripts/eval infrastructure, run `evaluate.sh doctor` and at least dry-run `evaluate.sh smoke` for each affected `--bench` before requesting review

---

## Review Instructions for Claude

When asked to review a PR (`/review`):

1. **1-sentence summary** + PR type (Task / Framework / Config / Scripts / Mixed).
2. **Run the relevant checklist** from above. Skip inapplicable sections.
3. **Report only real issues** — skip items that pass. For each issue: file:line, what's wrong, one-line fix.
4. **Score**: ✅ Approve / ⚠️ Approve with comments / ❌ Request changes.

Type-specific red flags to check first:
- *Task*: YAML labels ≠ Python labels? Still using `SkillManager` / `SyncCollectEnv` / `pre_*`?
- *Framework*: breaks existing tasks or eval client? `reset`/`soft_reset` miss new state?
- *Config*: YAML valid? shared config silently breaks other tasks?

When creating a PR (`gh pr create`):
- Title: `[Scope] type: description (≤70 chars)`
- Body: summary bullets, affected tasks, test plan (`bash scripts/evaluate.sh eval ...` or task-specific eval).

---

## Common Pitfalls

- `check_success` always `True` → garbage eval metrics
- YAML label `["target"]` but code uses `label="obj"` → silent mismatch
- Missing `soft_reset` override → state leak between episodes
- `step_lim = 200` on long-horizon tasks → truncation artifacts
- Camera config changes affecting all tasks without announcement
- Treating a launcher exit code alone as sweep completion → also require `_result.json` to cover the effective requested episode count
- Treating empty asset directories or a download check as proof of native layout coverage and GPU simulation
- Mixing MobileArena `joint` / `ee` checkpoint layouts or confusing profile names with saved `config_name` values
- Editing policy logic in RoboDojo instead of XPolicyLab
- Running policy `setup_eval_*` without CWD in the policy directory
- Treating submodule content as owned by this repo unless explicitly updating the gitlink pin
