# Benchmark scripts

Run the shared benchmark CLI from the repository root:

```bash
bash scripts/evaluate.sh <command> [options]
```

MobileArena is the default. Select `--bench SparkArena` for the seven Tianji
tabletop tasks; `--dataset` is an alias for `--bench`.
`bash scripts/MobileArena/mobile_arena.sh <command>` is a thin entry point for
the same CLI. See the [MobileArena guide](MobileArena/README.md) for tasks,
policy contracts, assets, and scoring.

See the [SparkArena guide](SparkArena/README.md) to import its assets and connect
to an existing Spark_0 server. External clients select it with
`--policy-name Spark_0`; model inference and weights stay on the server.
Set `EVAL_NUM_ENVS` to control parallel simulation for single tasks or sweeps.

## Public entry points

| Script | Purpose |
| --- | --- |
| [evaluate.sh](evaluate.sh) | Shared CLI: `doctor`, `eval`, `server`, `client`, `smoke`, `benchmark`, `summarize`, `tasks` |
| [install.sh](install.sh) | One-time environment setup (conda, Isaac Sim, submodules) |
| [init_assets.sh](init_assets.sh) | Download MobileArena assets, or import SparkArena with `--bench SparkArena --source /path/to/Assets` |
| [MobileArena/download_data.sh](MobileArena/download_data.sh) | Download `ridiculouslzx/MobileArena` from ModelScope using a token, parallel files and large-file range threads; [usage](MobileArena/README.md#download-the-modelscope-dataset) |
| [eval_policy.sh](eval_policy.sh) | Isaac Sim eval client (called by `evaluate.sh client` and XPolicyLab) |

The default profile is `mobile_arena`, with `config_name: mobile_arena` for layouts
and results. The simulator uses the `RoboDojo` Conda environment. An explicit
`--env-cfg` must declare the selected benchmark. SparkArena defaults to
`tianji_marvin_wuji`. The policy runs in its own
environment: `--policy-env` does not install dependencies or activate the
simulator for standalone `client` commands.

`tasks` lists the 24 runnable tasks; `doctor` performs preflight checks; `eval`
runs one task with a fresh server. `server` and `client` support split deployment
and batch workers when `--task` is omitted. `smoke` and `benchmark` support task
filters, GPU groups, logs, and resume. `summarize` applies the three-seed scoring.

## Download assets

Run the asset downloader after installing the simulator environment:

```bash
bash scripts/init_assets.sh                     # MobileArena (default)
bash scripts/init_assets.sh --bench MobileArena # explicit selection
bash scripts/init_assets.sh --help
```

The downloader selects MobileArena object, layout, M92UW robot, and room
packages, plus shared `Material/`, `Background/`, `Sensor/`, and `Traj/` assets
when present. Existing local assets and cache packages are retained.

The shared asset source has these environment overrides:

| Variable | Default | Meaning |
| --- | --- | --- |
| `HF_REPO_ID` | `RoboDojo-Benchmark/RoboDojo` | Hugging Face dataset repository |
| `HF_REPO_URL` | `https://huggingface.co/datasets/${HF_REPO_ID}` | Explicit Git repository URL; overrides the URL derived from the ID |
| `HF_REVISION` | `main` | Asset revision to download |

The source stores packages under `Assets/`. The selected revision must contain
all packages required by the requested benchmark; the downloader fails with a
missing-package error otherwise. In particular, use a revision containing
MobileArena assets.

Completion is checked against the source file list and sizes, and unresolved
Git LFS pointers are rejected. Empty directories do not count as downloaded
assets. This verifies the download; use `evaluate.sh doctor` to check a task's
native layout count and environment, then run a smoke test to verify simulation.

## Typical eval flow

```text
evaluate.sh eval
  -> scripts/internal/run_policy_eval.sh
    -> policy server (localhost) + sim client

Split / multi-machine:

evaluate.sh server  ->  scripts/internal/run_policy_server.sh  ->  policy server
evaluate.sh client  ->  scripts/eval_policy.sh  ->  src/eval_client/main.py
```

Use `bash scripts/evaluate.sh server --help` and
`bash scripts/evaluate.sh client --help` for split-evaluation options.
The [MobileArena split example](MobileArena/README.md#split-server-and-client) shows
matching benchmark, policy, profile, checkpoint label, action type, seed, and
port on both sides. `client` runs in the active simulator environment and does
not load or change the server's checkpoint; its `--ckpt` / `--action-type`
values label the results and must match the server.

Inspect tasks or validate the installation:

```bash
bash scripts/evaluate.sh tasks --bench MobileArena --check
bash scripts/evaluate.sh doctor --bench MobileArena \
  --task place_bread_in_oven --seed 0 --eval-num native --skip-policy
```

`doctor` checks the chosen task and seed, including the layout count needed for
the requested evaluation. Asset checks follow task-specific robot/scene
overrides and saved layout references: robot USD, URDF and required planner
configs, object USD and metadata, room USD, and configured background/material
files must exist, be nonempty, and contain downloaded data rather than Git LFS
pointers. USD-internal dependencies and actual asset loading still need a smoke test.

Checkpoint checks use XPolicyLab's runtime resolver. `--ckpt` accepts a short
name, full training-run directory name, absolute path, or path relative to the
policy directory; adapter-specific deploy/environment overrides take their
runtime precedence. Match `--action-type` and `--seed` to the evaluation
(`--action-type` defaults to `ee`). For example:

```bash
bash scripts/evaluate.sh doctor --bench MobileArena \
  --task place_bread_in_oven --policy-dir XPolicyLab/policy/Pi_05 \
  --ckpt /path/to/checkpoint --action-type joint --seed 0 --eval-num 1
```

This checks the checkpoint path, without loading weights or validating model
artifacts. The weight-free `demo_policy` does not require a checkpoint.
`doctor` does not launch a simulation. The `--skip-isaac`,
`--skip-conda`, and `--skip-policy` flags intentionally omit those checks; they
do not make an asset-free checkout ready to evaluate.

Select from the 24 tasks (eight base tasks and their `_l`/`_s` variants) using
`--only`, `--tasks-file`, or `--limit`.

## Batch execution and multi-GPU grouping

`evaluate.sh smoke` and `evaluate.sh benchmark` use the same scheduler.
Execution is sequential by default. `smoke` defaults to one episode
per task; `benchmark --eval-num native` uses each task's configured episode
count. Pass `--gpu-ids` to run task groups in parallel, with tasks inside each
group running sequentially. Separate `--policy-gpu-ids` and `--env-gpu-ids` lists
can assign policy servers and simulator clients to different devices.

Tasks are assigned round robin by count. Equal task counts do not guarantee
equal execution time.

The examples use the included [demo policy](../XPolicyLab/policy/demo_policy/README.md),
which holds the observed pose and requires no model checkpoint. Prepare its
`demo-policy` environment separately from the `RoboDojo` simulator environment
before an actual run. Batch scripts require Bash 4 or newer.

From the repository root, prepare the demo environment once:

```bash
conda create -n demo-policy python=3.11 -y
conda activate demo-policy
bash XPolicyLab/policy/demo_policy/install.sh
conda activate RoboDojo
```

Real policies need their own installation and compatible weights. Choose
`--action-type` explicitly: the CLI defaults to `ee`, while several included
MobileArena adapters require `joint`. See the
[compatibility list](MobileArena/README.md#policy-and-action-compatibility).

Preview all 24 MobileArena tasks split across four GPUs:

```bash
bash scripts/evaluate.sh benchmark --bench MobileArena \
  --policy-dir XPolicyLab/policy/demo_policy \
  --ckpt demo \
  --policy-env demo-policy \
  --action-type joint \
  --eval-num native \
  --gpu-ids 0,1,2,3 \
  --dry-run
```

Remove `--dry-run` to launch. A batch task passes validation only when its process
exits successfully and `_result.json` has a valid integer `eval_time` covering
the requested episode count, capped by that task's native count. For example,
MobileArena `--eval-num 1` requires one completed episode, while `--eval-num native`
requires 25 for every MobileArena task, including base tasks and `_l`/`_s` variants,
as configured in `task/MobileArena/config/_task.yml`. A partial native evaluation is marked `FAIL`
and is not skipped by resume. `PASS` describes evaluation completion, not policy
task success.
Single-task `eval` and `client` also exit nonzero when the simulator exhausts
layouts before reaching the effective episode count, retaining any resume
manifest. Task failures count as completed episodes; skipped or unstable layouts
do not.

`--gpu-ids` distributes tasks across processes. In contrast, `EVAL_NUM_ENVS`
controls simultaneous environments inside each simulator process:

```bash
EVAL_NUM_ENVS=2 bash scripts/evaluate.sh eval --bench MobileArena \
  --policy-dir XPolicyLab/policy/Pi_05 --policy-env <POLICY_ENV> \
  --task place_bread_in_oven --ckpt <CHECKPOINT> \
  --action-type joint --eval-num 2
```

MobileArena defaults to one environment. RoboDojo uses `num_envs` to select the
single- or multi-environment rollout and does not read or override XPolicyLab's
`eval_batch` setting. XPolicyLab owns model inference batching. Multi-environment
rollouts require the adapter's `eval_one_episode_batch` entry point and model
methods that retain observation/action state by environment index; those methods
can evaluate observations together or one at a time. Choose an environment count
that fits the simulator and model resources.

### Logs and resume

Batch logs and status files default to `smoke_results/MobileArena/`.
Set `--run-id NAME` for a repeatable sweep identity, then rerun with `--resume`
to skip completed tasks. Keep the same execution mode (`eval` or `client`),
benchmark, profile, policy, checkpoint, action type, seed, and episode count.
Alternatively, point `--summary PATH` at the original summary. Server mode
starts persistent servers and does not support `--resume`.

On Ctrl-C or SIGTERM, parallel sweeps save finished workers' results before
exiting. Resume skips those completed tasks and reruns unfinished ones.

For independent comparisons, restart externally managed policy servers when
their sampling RNG must start from the same state; an episode-level reset does
not necessarily reset the model's RNG.

## Result summaries

```bash
bash scripts/evaluate.sh summarize --bench MobileArena
```

The command runs [MobileArena/summarize_result.py](MobileArena/summarize_result.py)
and writes `eval_result/MobileArena/_summary.md` by default.

When an adapter accepts a checkpoint path, pass the original path to `--ckpt`.
Result directory labels containing path separators are percent-encoded with a
`urlencoded:` prefix so the experiment remains in one directory. Model loading
still receives the original checkpoint argument; MobileArena summary filters also
accept that original value, including its full path.
Encoded labels exceeding 200 bytes are rejected; use a shorter checkpoint name
or path. This keeps result labels reversible without adding metadata to `_result.json`.

See the
[MobileArena guide](MobileArena/README.md#result-summaries) for its three-seed aggregation
and result filters.

For MobileArena, `--env-cfg` on `summarize` filters the saved `config_name`;
launch commands use it to select the profile stem. Both values are `mobile_arena`
in the default `env_cfg/mobile_arena.yml` profile.
MobileArena summaries select the run containing the most recently modified
`_result.json` across all directories for each decoded experiment/task/seed,
including plain, `urlencoded:`, and legacy `sha256:` labels that identify the
same experiment. Ties use the run-directory name, then the experiment-directory
name. An incomplete or invalid
latest result remains incomplete instead of falling back to an older result.
Full scoring covers 24 tasks and seeds `0,1,2` (72 task/seed cells), requiring
25 valid episode records per task at each seed, as configured in
`task/MobileArena/config/_task.yml`. All MobileArena `±` values use population standard
deviations across seeds, in both the overview and detailed task tables.

## CPU validation

With Python 3.11+, PyYAML, NumPy, and Bash 4+:

```bash
python scripts/internal/task_inventory.py --format json --check
bash scripts/evaluate.sh doctor --skip-isaac --skip-conda --skip-policy
bash -n scripts/evaluate.sh scripts/eval_policy.sh
```

Inventory checks and `--dry-run` validate configuration and command routing.
They do not validate rendering, physics, Pink IK, or learned checkpoint inference.
Run `doctor` and an actual `smoke` on the target machine before a full benchmark;
keep smoke completion separate from policy success metrics.

## Internal (`internal/`)

These scripts are called by `evaluate.sh` or policy utilities.

| File | Purpose |
| --- | --- |
| [benchmark_profile.py](internal/benchmark_profile.py) | Benchmark metadata, environment defaults, and result configuration |
| [task_inventory.py](internal/task_inventory.py) | MobileArena task discovery and config/class consistency |
| [verify_install.sh](internal/verify_install.sh) | `evaluate.sh doctor` for the selected benchmark |
| [check_assets.py](internal/check_assets.py) | Static checks of selected robot, scene, object, and layout asset files |
| [check_policy_checkpoint.py](internal/check_policy_checkpoint.py) | Checkpoint path preflight using XPolicyLab's runtime resolver |
| [smoke_all_tasks.sh](internal/smoke_all_tasks.sh) | Shared `smoke` / `benchmark` runner |
| [benchmark_groups.py](internal/benchmark_groups.py) | Round-robin task assignment across GPU workers |

## Docker

Container install and smoke tests live under [../docker/](../docker/).

## Policy-specific scripts

Training, data prep, and per-policy `eval.sh` live in [../XPolicyLab/policy/](../XPolicyLab/policy/) (submodule).
