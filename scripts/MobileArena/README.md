# MobileArena evaluation scripts

Use the shared benchmark CLI from the repository root:

```bash
bash scripts/evaluate.sh <command> --bench MobileArena [options]
```

Commands are `tasks`, `doctor`, `eval`, `server`, `client`, `smoke`, `benchmark`,
and `summarize`. MobileArena is the only supported benchmark and the default,
so `--bench MobileArena` is optional. The default environment profile is
`mobile_arena`; the simulator still uses the shared `RoboDojo` Conda environment.
`--dataset MobileArena` is a compatibility alias for `--bench MobileArena`.

Task discovery, validation, single-task launch, and batch scheduling use
`scripts/evaluate.sh`. See the [CLI guide](../README.md) for the workflow.

Existing `bash scripts/MobileArena/mobile_arena.sh <command>` calls remain supported as a
thin alias for `evaluate.sh <command> --bench MobileArena`.

## Installation and assets

Use the [root installation steps](../../README.md#mobilearena-quick-start) on a
Linux NVIDIA GPU machine. MobileArena runs in the `RoboDojo` simulator
environment, including the shared Pinocchio/Pink dependencies. The policy has
a separate environment installed according to its adapter README.

Download MobileArena packages and recorded layouts from the repository root:

```bash
bash scripts/init_assets.sh --bench MobileArena
# Omitting --bench also selects MobileArena.
```

The downloader uses the shared Hugging Face asset source. `HF_REPO_ID`,
`HF_REPO_URL`, and `HF_REVISION` select the repository and revision; see the
[asset options](../README.md#download-assets). The selected revision must
contain the complete MobileArena packages below. If any required package is
absent, the downloader reports an error.

```text
Assets/
  Object/MobileArena/                 task objects
  Room/MobileArena/                   kitchen and living-room scenes
  Robots/m92uw/                   complete M92UW robot/URDF/USD/IK package
  Material/                      shared materials
  Background/                    shared environment lighting textures
  Eval_Layout/MobileArena/mobile_arena/
    0/<task_name>_<index>.json
    1/<task_name>_<index>.json
    2/<task_name>_<index>.json
```

Keep referenced files within those packages intact. The downloader checks
source file lists and sizes and rejects unresolved Git LFS pointers; empty
directories do not pass. Download verification does not check GPU simulation
or the native layout counts. A full evaluation needs at least 25 usable layouts
per task for all 24 tasks, including base tasks and `_l`/`_s` variants, for each
seed `0`, `1`, and `2`, as configured in `task/MobileArena/config/_task.yml`.
Layout indices are sorted numerically. Both the `mobile_arena` profile
and its layout/result `config_name` are `mobile_arena`.

For a weight-free interface check, create the demo environment from the root:

```bash
conda create -n demo-policy python=3.11 -y
conda activate demo-policy
bash XPolicyLab/policy/demo_policy/install.sh
conda activate RoboDojo
```

## Download the ModelScope dataset

The token-authenticated downloader uses
[`ridiculouslzx/MobileArena`](https://modelscope.cn/datasets/ridiculouslzx/MobileArena/tree/master)
at revision `master`. Install the optional
[ModelScope Hub SDK](https://github.com/modelscope/modelscope_hub) in the Python
environment used to run the downloader:

```bash
python3 -m pip install -U 'modelscope-hub>=0.5.0'
bash scripts/MobileArena/download_data.sh --local-dir /path/to/MobileArena
```

The command prompts for a token with hidden input in an interactive terminal.
For a noninteractive run, set `MODELSCOPE_API_TOKEN` or supply
`--token-file /path/to/token.txt`. `--token TOKEN` is also supported, but the
prompt avoids putting the token in shell history. The script passes the token
in memory and does not save a ModelScope login configuration.

For example, read a token once and inspect the file selection before downloading:

```bash
read -rsp 'ModelScope token: ' MODELSCOPE_API_TOKEN; echo
export MODELSCOPE_API_TOKEN
bash scripts/MobileArena/download_data.sh --list
bash scripts/MobileArena/download_data.sh --local-dir /path/to/MobileArena \
  --max-workers 8 --file-workers 8
unset MODELSCOPE_API_TOKEN
```

| Option | Meaning |
| --- | --- |
| `--local-dir PATH` | Destination; default is this repository's `data/MobileArena/` regardless of the working directory |
| `--repo-id ID`, `--revision REV` | Override the dataset repository or revision |
| `--include GLOB` | Include matching remote paths; repeatable, default includes every file |
| `--exclude GLOB` | Exclude matching paths; repeatable, overrides includes |
| `--max-workers N` | Number of concurrent file downloads, default `8` |
| `--file-workers N` | Range download threads per large file, `1`–`16`, default `8`; `1` disables splitting |
| `--parallel-threshold-mb N` | Split files larger than this many MiB, default `64` |
| `--list` | List selected paths and sizes without downloading file contents |
| `--force` | Redownload selected files even if cached |

Quote glob patterns, for example `--include '*.hdf5'`. Patterns match the full
remote path, `*` also matches `/`, and a trailing `/` selects that directory's
contents. Use `--list` to inspect the repository before choosing filters. An
empty selection is an error.

Downloads use two levels of concurrency: up to eight files at once, with up
to eight range requests per large file (up to 64 download connections by
default). Files use fixed 64 MiB parts, so changing the thread counts preserves
existing range checkpoints. Smaller files use one connection. The script
sets the SDK's parallel worker, threshold and part-size settings before
importing it; these CLI settings take precedence over the corresponding SDK
environment variables. The SDK falls back to a single stream if the server
does not support range requests. For lower bandwidth or throttling, reduce
`--max-workers` or `--file-workers`. Actual throughput depends on the server
and network.

To accelerate an existing download, stop the old process before starting the
updated script with the **same** local directory, repository, revision and
filters. Leave out `--force` to reuse verified completed files. Existing
nonempty `.incomplete` files keep single-stream resume so the bytes already
downloaded are preserved; newly started large files use range threads. Keep
partial files in place. Range downloads report completed-file counts rather
than the single-stream byte progress bars.

The download preserves the remote directory structure. It does not unpack
archives or install files into the simulator's `Assets/` directory; use the
asset installation steps above for simulation. Re-running the same command
lets the SDK reuse checksum-verified cached files and resume interrupted
transfers. Files without a remote checksum are downloaded again. Use a new
destination when switching repositories or revisions to avoid reusing partial
transfers from a different source. Completion
requires every selected file to exist with its advertised size; remote SHA-256
checksums are passed to the SDK when available. A failed or incomplete download
exits with a nonzero status.

## Policy and action compatibility

The simulation accepts joint targets or base-frame end-effector poses, plus
planar base velocity. Set `--action-type` to the adapter/checkpoint contract;
the shared CLI defaults to `ee` and does not convert one model layout into another.

| Adapter | `--action-type` | MobileArena checkpoint contract |
| --- | --- | --- |
| [FastWAM](../../XPolicyLab/policy/FastWAM/README.md) | `joint` | MobileArena 45D joint layout and matching dataset statistics |
| [MM_ABC](../../XPolicyLab/policy/MM_ABC/README.md) | `joint` | MobileArena M92UW `m75` or `c80`, selected from checkpoint metadata |
| [MoPA](../../XPolicyLab/policy/MoPA/README.md) | `joint` | MobileArena `mopa75` contract used by the included deploy configuration |
| [OpenWAM](../../XPolicyLab/policy/OpenWAM/README.md) | `ee` | MobileArena EE45 mapping in an 80D OpenWAM checkpoint, with `mobile_arena_ee` data config and matching normalization keys |
| [Pi_05](../../XPolicyLab/policy/Pi_05/README.md) | `joint` | Current 45D MobileArena joint layout; old 75D checkpoints are incompatible |
| [Xiaomi_Robotics_1](../../XPolicyLab/policy/Xiaomi_Robotics_1/README.md) | `ee` | MobileArena official-state 60D XR1 checkpoint, with 39 active action dimensions |
| [demo_policy](../../XPolicyLab/policy/demo_policy/README.md) | `joint` or `ee` | No weights; holds observed posture and commands zero base velocity |

The 45D joint vector contains 14 arm joints, 24 hand joints, 2 waist joints,
2 leg joints, and base `[vx, vy, wz]`. OpenWAM's EE45 vector instead contains
two 9D position/rotation representations, 24 hand joints, and the same three
base controls. Equal vector length does not make these layouts interchangeable.

Runtime EE poses use `[x,y,z,qw,qx,qy,qz]` in the robot's `base_link` frame;
whole-body Pink IK controls the arm/waist/leg joints in EE mode. Root velocities
are world-frame commands; only `vx`, `vy`, and `wz` are applied. Vertical motion
and roll/pitch rates are outside the mobile-base action contract. The learned
adapters use head and two wrist RGB views; the simulator also exposes shared
joint groups and base-frame poses in its observation dictionary.

## Inspect and run tasks

```bash
bash scripts/evaluate.sh tasks --bench MobileArena --check
bash scripts/evaluate.sh doctor --bench MobileArena \
  --task place_kettle_on_tv_stand --seed 0 --eval-num 1 \
  --policy-dir XPolicyLab/policy/demo_policy --policy-env demo-policy
bash scripts/evaluate.sh eval --bench MobileArena \
  --task place_kettle_on_tv_stand \
  --policy-dir XPolicyLab/policy/demo_policy \
  --ckpt demo --policy-env demo-policy \
  --action-type joint --seed 0 --eval-num 1 \
  --dry-run
```

`doctor --seed N --eval-num NUM|native` checks the selected task's configuration,
assets, and saved layouts for that seed/count. It checks one task per invocation;
repeat it for the tasks and seeds you intend to run. Asset files must be
nonempty and resolved from Git LFS, including the selected
M92UW, room, object and background files. For a learned policy, add `--ckpt
NAME_OR_PATH --action-type joint|ee` with the same action type as evaluation.
Checkpoint paths use the runtime resolver; this does not load model weights.
`--skip-isaac`, `--skip-conda`, or `--skip-policy` omit those checks only.

After preflight succeeds, remove `--dry-run` to evaluate. Demo completion checks
the interface; it is not evidence that a trained policy can solve the task.
Both `eval` and `client` return a failure status if the available layouts are
exhausted before the requested valid episode count is reached. Any existing
resume manifest is retained. A completed episode with task failure still counts;
an unstable or skipped layout does not.

## Split server and client

For a single task, start a server in one terminal or machine:

```bash
bash scripts/evaluate.sh server --bench MobileArena \
  --policy-dir XPolicyLab/policy/demo_policy --policy-env demo-policy \
  --task place_kettle_on_tv_stand --ckpt demo \
  --env-cfg mobile_arena --action-type joint --seed 0 \
  --policy-gpu 0 --bind-host 0.0.0.0 --policy-port 6000
```

In a second terminal, activate the simulator environment and connect:

```bash
conda activate RoboDojo
bash scripts/evaluate.sh client --bench MobileArena \
  --policy-name demo_policy --task place_kettle_on_tv_stand --ckpt demo \
  --env-cfg mobile_arena --action-type joint --seed 0 \
  --policy-host 127.0.0.1 --policy-port 6000 --env-gpu 0 --eval-num 1
```

Replace `127.0.0.1` with the policy machine's reachable address for remote
deployment. Keep the benchmark, policy, task, profile, checkpoint label, action
type, and seed aligned. The client labels its results with `--ckpt` and
`--action-type`; these options do not change the already running server.

Omit `--task` to use batch server/client mode. For two workers on one machine:

```bash
# Policy terminal: one server per port/GPU; remains running.
bash scripts/evaluate.sh server --bench MobileArena \
  --policy-dir XPolicyLab/policy/demo_policy --policy-env demo-policy \
  --ckpt demo --env-cfg mobile_arena --action-type joint --seed 0 \
  --policy-gpu-ids 0,1 --policy-port 6000,6001

# Simulator terminal:
conda activate RoboDojo
bash scripts/evaluate.sh client --bench MobileArena \
  --policy-name demo_policy --ckpt demo \
  --env-cfg mobile_arena --action-type joint --seed 0 \
  --env-gpu-ids 0,1 --policy-host 127.0.0.1 --policy-port 6000,6001 \
  --eval-num 1
```

Ports correspond to worker order; their count must match the client GPU groups.
One host is reused for all ports, or pass one host per worker. Use matching task
filters and worker counts when splitting a sweep. Server mode has no result
completion state and does not support `--resume`.

The task inventory includes eight base tasks and their `_l` (language) and `_s`
(spatial) variants, for 24 tasks in total. Use the full task name, for example
`--task place_kettle_on_tv_stand_l`. Each variant uses its own eval layouts under
`Assets/Eval_Layout/MobileArena/mobile_arena/<seed>/<task>_<layout_id>.json`.

For `_l` tasks, the instruction is the exact nonempty `instruction` saved in each
layout; missing text is an error. Instructions refresh when the next layout
batch loads. For `place_bread_in_oven_s`, the pickup surface is taken from that
layout's bread object (`kitchen_island`, `cutting_board`, or `table`), so an old
generic or mismatched instruction is rebuilt to name the actual surface.
`throw_away_trash_s` refreshes the package name for each layout batch;
`put_object_in_sink_s` retains the source task's generic instruction.

## Batch evaluation

```bash
bash scripts/evaluate.sh benchmark --bench MobileArena \
  --policy-dir XPolicyLab/policy/Pi_05 \
  --ckpt <CHECKPOINT> \
  --policy-env <POLICY_ENV> \
  --action-type joint \
  --seed 0 \
  --eval-num native \
  --dry-run
```

`smoke` defaults to one episode per task. `benchmark --eval-num native` uses the
episode counts in `task/MobileArena/config/_task.yml` (25 for each of the 24 tasks,
including base tasks and `_l`/`_s` variants), even if `EVAL_NUM` is set in the
parent shell. Repeat the command with `--seed 1` and `--seed 2` for the
three-seed summary.

Select tasks with `--only task_a,task_b`, `--tasks-file PATH`, or `--limit`.
MobileArena has 24 tasks and no capability dimensions. Execution is sequential
by default. To assign tasks across GPUs, add `--gpu-ids 0,1,2,3`. MobileArena has
no measured runtime weights yet, so the shared scheduler assigns tasks round
robin by task count. This does not guarantee equal runtimes across workers.

GPU grouping assigns different tasks to separate simulator processes. To change
the number of simultaneous environments within each process, set
`EVAL_NUM_ENVS=2` (or another supported count) before the command. MobileArena's
profile defaults to one environment. RoboDojo selects the rollout from this
environment count independently of XPolicyLab's `eval_batch` setting. The policy
adapter must support the multi-environment interface and keep each environment's
state separate; model inference batching is handled within XPolicyLab. The
usable environment count still depends on available memory and runtime validation.

In batch `eval` mode, each task starts a fresh policy server. Batch `client`
mode connects to external servers via `--policy-host` / `--policy-port`.
Restart external servers before independent comparisons when matching policy
sampling RNG state is required; a policy's episode reset may not reset its RNG.

Batch status summaries and logs default to `smoke_results/MobileArena/`. To resume
an interrupted sweep, reuse `--run-id` (or `--summary`) with `--resume` and the
same profile, policy, checkpoint, action type, seed, and episode count. A task
must also use the same execution mode (`eval` or `client`) when resumed. A task
passes batch validation only if its process exits successfully and its saved
result contains an integer `eval_time` reaching `min(requested, native)`:
one episode for a one-episode smoke, or 25 for every native task, including
base tasks and `_l`/`_s` variants. Partial runs are marked `FAIL` and are not
skipped on resume. This status measures
evaluation completion, separately from the policy's task success rate.

Parallel sweeps save completed workers' results when interrupted by Ctrl-C or
SIGTERM, so resuming the same sweep skips completed tasks.

## Result summaries

The environment profile is named `mobile_arena`. Layouts and saved results use its
`config_name`, also `mobile_arena`:

```text
eval_result/MobileArena/<task>/<policy>/mobile_arena/<seed>_ckpt_name=...,action_type=.../<run_id>/_result.json
```

If `--ckpt` contains a path, the run label after `<seed>_` uses a `urlencoded:`
prefix and percent encoding to keep it in one directory. Pass the original
checkpoint path to both evaluation and `summarize --ckpt`; the summary decodes
the label for filtering and display. Ordinary checkpoint names keep the layout
shown above.
Encoded labels exceeding 200 bytes are rejected; use a shorter checkpoint name
or path. No additional checkpoint metadata is written to `_result.json`.

```bash
bash scripts/evaluate.sh summarize --bench MobileArena
bash scripts/evaluate.sh summarize --bench MobileArena \
  --ckpt <CHECKPOINT> --action-type joint --env-cfg mobile_arena
```

The summary script can also run directly:

```bash
python scripts/MobileArena/summarize_result.py \
  --root eval_result/MobileArena --output /tmp/mobile_arena_summary.md
```

`--root` selects the result root containing task directories; otherwise
`MOBILE_ARENA_EVAL_ROOT` or `eval_result/MobileArena` is used. `--output` overrides the
default `<root>/_summary.md`. For summary filtering, `--env-cfg` is the result
**config_name directory**, set to `mobile_arena` in `env_cfg/mobile_arena.yml`. For a custom
profile, filter by the `config_name` saved in its result directories.

Each result configuration and run label, including checkpoint and action type,
forms a separate experiment. Summaries report success rate and score for all
24 tasks and seeds 0, 1, 2, with progress out of 72 task/seed cells. A cell
requires the task's native episode count (25 for all 24 tasks, as configured in
`task/MobileArena/config/_task.yml`), and only the first such records are scored.
The run with the newest `_result.json` modification time is selected across all
directories for the same decoded experiment/task/seed, including plain,
`urlencoded:`, and legacy `sha256:` labels that identify the same experiment.
Ties use the run-directory name, then the experiment-directory name. Directories
without a result file are ignored. An incomplete or
invalid latest result does not fall back to an older completed result.
Incomplete cells stay blank, and invalid files are reported in the terminal.
Smoke results with one episode do not fill a full benchmark cell. All `±`
values use population standard deviations across seeds, in both the overview
and detailed task tables.

## Validation scope

CPU checks can run without Isaac Sim or model weights, with Python 3.11+,
PyYAML, NumPy, and Bash 4+:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Coverage includes task routing, launch arguments, batch completion/resume,
summary selection, layout ordering, and selected policy action-layout conversions.
These checks do not execute physics, render assets, run Pink IK, or load learned
checkpoints. The current review has not run those GPU-dependent paths. A real
one-episode smoke on the configured evaluation machine is still required before
claiming the full simulation path works; policy quality requires all 24 tasks
at their native episode counts across all three seeds.
