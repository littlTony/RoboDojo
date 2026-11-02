# SparkArena environment evaluation

SparkArena adds seven tabletop tasks for the Tianji Marvin robot with WujiHand2
hands. It shares the current `scripts/evaluate.sh` CLI with MobileArena;
MobileArena remains the default. Select `--bench SparkArena` (or the alias
`--dataset SparkArena`) explicitly.

This checkout supplies the simulation environment and WebSocket client. The
Spark_0 model, its dependencies and checkpoint stay on the existing policy
server. The fork's XPolicyLab and IsaacLab submodule revisions are retained.
The simulator needs XPolicyLab's shared WebSocket transport, but does not need a
local `policy/Spark_0` model adapter to use `client` mode.

## Installation and assets

Initialize the recorded submodules and use the shared simulator installer:

```bash
git submodule update --init --recursive
bash scripts/install.sh --install
conda activate RoboDojo
bash scripts/init_assets.sh --bench SparkArena --source /path/to/Assets
```

`--source` is an existing **Assets directory**, containing these paths:

| Path under Assets | Contents |
| --- | --- |
| `Robots/tianji_marvin_wuji/` | Left/right USDs, URDFs, meshes, camera textures, robot descriptions and Curobo templates |
| `Object/SparkArena/` | Rigid, articulated and geometry objects, including metadata |
| `Eval_Layout/SparkArena/tianji_marvin_wuji/` | Seed directories containing saved task layouts |
| `Room/Simple_Room_nolight/` | Room and dependencies |
| `Material/material_0116/`, `Material/material_0122/` | Table and ground materials |
| `Background/brown_photostudio_02_4k.hdr` | Environment lighting |

Import validates these paths before copying, preserves existing asset files and
source content, and regenerates local Curobo paths and relative camera texture
references. Run `python utils/update_embodiment_config_path.py` after relocating
the checkout. Run preparation inside the simulator environment; the shared
requirements include `usd-core==26.3` for USD editing without starting Kit.
SparkArena import requires `--source`; MobileArena keeps its existing Git/LFS
download path. Assets, generated robot paths and checkpoints are not committed.

Inspect tasks and validate the selected seed and episode count:

```bash
bash scripts/evaluate.sh tasks --bench SparkArena --check
bash scripts/evaluate.sh doctor --bench SparkArena \
  --task stack_bowls --seed 0 --eval-num native \
  --skip-isaac --skip-conda --skip-policy
```

Omit the skip flags for simulator/Conda validation. Policy checks apply to local
model deployments; an external Spark_0 server owns its model and weights.

## Connect to an existing Spark_0 server

Replace `SERVER_HOST` and `6001` with the existing service's reachable address.
Run inside the activated simulator environment:

```bash
EVAL_NUM_ENVS=5 bash scripts/evaluate.sh client --bench SparkArena \
  --task click_mouse --policy-name Spark_0 \
  --policy-host SERVER_HOST --policy-port 6001 \
  --eval-batch true --eval-num native --seed 0 --env-gpu 0 \
  --ckpt steps40000 --action-type ee
```

`--policy-name` labels the remote policy and result directory. It does not load
model code locally. `--eval-batch true` explicitly declares that this server
accepts batches; use `false` for a server that accepts one environment. If
omitted, the client retains the existing local `deploy.yml` batch setting,
falling back to a single environment when no local adapter is installed.
`--ckpt` is the result label in client mode; it does not change the server's
loaded checkpoint. `EVAL_NUM_ENVS=5` sets the parallel environment count for this
run, while `--eval-num native` requests 50 valid episodes per task.

Omit `--task` to run all seven tasks sequentially against the same server:

```bash
EVAL_NUM_ENVS=5 bash scripts/evaluate.sh client --bench SparkArena \
  --policy-name Spark_0 --policy-host SERVER_HOST --policy-port 6001 \
  --eval-batch true --eval-num native --seed 0 --env-gpu 0 \
  --ckpt steps40000 --action-type ee
```

Use `--only click_mouse,stack_bowls` for a subset. Add `--dry-run` to inspect
commands without starting simulation or contacting the service. Client sweeps
reuse the shared task grouping, progress, logs and resume handling. Starting a
new local model with `eval` or `server` requires a separately installed policy
adapter; those commands do not install Spark_0.

## Tasks and scoring

| Task | Action limit | Score tiers / 100 |
| --- | ---: | --- |
| `click_mouse` | 400 | 0 / 100 |
| `collect_objects` | 600 | 0 / 15 / 40 / 100 |
| `dual_bottles_pick` | 400 | 0 / 30 / 100 |
| `hammer_beat` | 300 | 0 / 100 |
| `put_food_in_microwave` | 450 | 0 / 100 |
| `retrieve_gap` | 800 | 0 / 100 |
| `stack_bowls` | 700 | 0 / 15 / 100 |

Task instructions, object placement, success conditions and partial-score rules
are retained from the previously validated SparkArena adapter. Full success is
recorded as 100 points. Unstable scenes are excluded from the valid episode
count and replaced by spare layouts. Microwave translucency is disabled only
for that task; its constructor creates the optional `sim.render` node first.

## Configuration and control

The `tianji_marvin_wuji` profile selects the shared `sim_config`, Tianji tabletop,
robot and camera YAMLs. Common simulation defaults are `dt=0.004`,
`decimation=1` and ten parallel environments. SparkArena task metadata sets
`render_interval=5`; `data_source: teleop` enables PhysX stabilization through
the normal configuration loader. No separate teleop simulation profile is
required. Observations are collected at 25 Hz with two capture render passes.

The two arms each use seven arm joints and 20 independent hand joints. Joint
actions use left/right arm and hand joint states. EE actions use left/right EE
poses `[x, y, z, qw, qx, qy, qz]` plus hand joint states. Direct Curobo IK,
authored-joint reset, zero velocity targets, hand-open scoring and camera
extrinsics conversion remain scoped to `SparkArenaEvalEnv`. Shared EvalEnv
continues to handle interpolation, model communication, videos and recovery.
The head and wrist camera calibrations are private to the Tianji profile;
MobileArena's profiles and navigation evaluator remain in place.

## Results

```text
task/SparkArena/tasks/<task>.py
task/SparkArena/config/<task>.yml
eval_result/SparkArena/<task>/<policy>/<config_name>/<seed>_<run_info>/<run_id>/_result.json
smoke_results/SparkArena/<run_id>/
```

The current result-label encoding and resume paths are retained. Summarize
complete task results with:

```bash
bash scripts/evaluate.sh summarize --bench SparkArena \
  --env-cfg tianji_marvin_wuji --ckpt steps40000 --action-type ee
```

The default summary seed is 0; repeat `--seed` to select additional seeds.
Experiments, checkpoint labels and profiles remain separate. A task contributes
to the mean only after its native episode count is complete. The latest run is
selected by modification time; an incomplete or invalid latest run does not
fall back to an older complete result. Success rate and partial completion
score are reported separately. MobileArena keeps its own summary rules.
