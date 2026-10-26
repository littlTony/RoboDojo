# SparkArena entry points

SparkArena uses the shared public CLI:

```bash
bash scripts/robodojo.sh tasks --dataset SparkArena --format markdown
bash scripts/robodojo.sh dimensions --dataset SparkArena
bash scripts/robodojo.sh summarize --dataset SparkArena
```

Reuse the existing `--dataset` and `--env-cfg` arguments. Evaluation commands
pass `--dataset SparkArena --env-cfg tianji_marvin_wuji` explicitly; the original
RoboDojo defaults remain in place. XPolicyLab's `--bench_name` and the shared
client's `--dataset_name` are consumed by `eval_policy.sh` and forwarded as
`--bench_name` to Python, which checks the profile's benchmark before launching.

## Evaluation support

SparkArena provides its namespace, seven tasks and their YAMLs,
launch arguments, benchmark-scoped results, Spark evaluation wrapper, and Tianji
robot registration/configuration, action control, camera/observation compatibility,
capture timing, saved-layout replay, and physical-state resets. The wrapper
shares the public EvalEnv lifecycle, policy connection, videos, and
crash/resume handling. Tasks use the existing public TaskEnv and RewardManager;
collection-only expert stubs, soft resets, and the pre_click_mouse environment
are omitted. Asset import, portable robot paths, installation checks, and the
Spark_0 XPolicyLab revision are included.

The profile and task-level overrides use the current development configuration
names. Matching scene and camera YAMLs are included:

| Profile field | Config filename |
| --- | --- |
| `config.sim` | `sim/sim_config.yml` |
| `config.scene` | `scene/tianji_tabletop.yml` |
| `config.robot` | `robot/tianji_marvin_wuji.yml` |
| `config.camera` | `camera/tianji_camera.yml` |

`src/eval_client/main.py` retains configuration assembly and selects the ordinary,
navigation, or Spark evaluator from `eval_cfg.benchmark`. Heavy simulation modules
are loaded after Kit starts. `scripts/eval_policy.sh` passes `--enable_cameras`
for every benchmark. All benchmarks keep the default viewport rendering in
headless mode before the app starts. Configuration assembly loads these exact
filenames through the existing YAML loader. The profile directly uses the public
`observation.robot/vision` and `world_ee_state` settings. Spark adds the camera
pose field required by the policy to the existing observation payload.

The Spark wrapper keeps its camera template, two capture passes, env-local
camera extrinsics, direct Curobo IK, hand-open check, zero-velocity targets and
authored-joint reset local to its environment instance. Spark reuses the public
legacy stability check, matching the evaluation-server rules.

The task inventory scans `task/SparkArena/tasks` directly and checks each class
against its matching YAML. This keeps `stack_bowls` independent of the RoboDojo
task with the same name. All seven pass `tasks --dataset SparkArena --check` and
appear with `--only-runnable`. These checks validate definitions; `doctor` also
checks runtime components.

## Task rules

Task instructions, placement ranges, asset candidates, thresholds, step limits,
and score registration follow the current development task definitions.

| Task | Action limit | Success conditions | Score tiers |
| --- | ---: | --- | --- |
| `click_mouse` | 400 | Mouse on pad and left button ratio > 0.75; then mouse still on pad and arms home | 0 / 100 |
| `collect_objects` | 600 | Three targets inside the basket, each with positive depth overlap; arms home | 0 / 15 / 40 / 100 |
| `dual_bottles_pick` | 400 | Both bottles lifted > 0.20 m from their initial heights | 0 / 30 / 100 |
| `hammer_beat` | 300 | Hammer lifted > 0.02 m; active head XY distance < 0.025 m from cube; either X axis within 45 degrees of up; head 0.01 m below to 0.04 m above cube support | 0 / 100 |
| `put_food_in_microwave` | 450 | Bread inside microwave support circle and 0.015–0.18 m above support; door ratio > 0.9; arms home | 0 / 100 |
| `retrieve_gap` | 800 | Both cubes moved > 0.01 m and upright within 15 degrees; garage 0.005–0.15 m above cushion, XY distance < 0.05 m and upright; arms home | 0 / 100 |
| `stack_bowls` | 700 | Three bowls stacked, each within 45 degrees of up, lowest within 7 degrees; adjacent XY offsets at most 0.04 m; arms home | 0 / 15 / 100 |

Count scoring gives 15/40 for one/two collected objects and 30 for one lifted
bottle. Bowl process scoring advances through two-bowl (15) and three-bowl
(100) states with open hands (threshold 0.8). The public reward engine gates the
100 tier on full success, including any required return home. The evaluator
always records a successful episode as 100 points. Other tasks use binary
success scoring. Microwave translucency remains disabled only for that task.

## Robot and action control

`tianji_marvin_wuji_left` and `tianji_marvin_wuji_right` are independent
articulations registered with the existing RobotManager. Their shared description
module loads the existing asset-side `left_robot_config.yml` and
`right_robot_config.yml`. Side-specific Isaac configs preserve the source arm
home poses, 20 hand joints, actuator gains, and fixed root settings. Evaluation
explicitly uses Curobo. The teleoperation send-list fields are omitted from the
evaluation robot YAML.

| Action mode | Arm fields (each 7 values) | Hand fields (each 20 values) |
| --- | --- | --- |
| Joint | `left_arm_joint_state`, `right_arm_joint_state` | `left_ee_joint_state`, `right_ee_joint_state` |
| Cartesian | `left_ee_pose`, `right_ee_pose` | `left_ee_joint_state`, `right_ee_joint_state` |

Both modes require all four fields. Joint targets are radians; EE targets use
metres and `[x, y, z, qw, qx, qy, qz]` in each environment's world frame. Hand
targets are 20 independent joint positions. Articulation joint indices use the
same `find_joints` behavior as the development version. Action validation reuses
the existing evaluator's key, type, dimension, and action-mode checks.

The existing interpolation and control queues are reused: arm targets interpolate
over the first 80% of an observation interval, then hold; hand targets apply
directly throughout. Initialization/reset sets joint velocity targets to zero;
Spark's component-control path updates hand positions while retaining that zero
hand velocity target. The existing MiVerse named-joint `qpos/qvel` path and
scalar gripper mimic control are retained.

Direct Curobo IK starts from measured seven-joint arm positions and uses the
env-local base-link pose, preserving scalar-first quaternions. The public planner
already selects active arm joints by name when a full solution includes hand
joints. A failed solve keeps the previous arm target while allowing hand control.
Reset writes all authored joint positions and clears velocity targets.

Asset setup includes URDFs, USDs, meshes and asset-side descriptions. The existing
path generator produces `curobo_left.yml` and `curobo_right.yml` for this checkout,
correcting the source templates' old `tianji_wujihand2` directory name. The actual
URDF filenames retain that name.

## Cameras, observations and timing

The public ObsManager and CameraManager are reused. RobotManager already names
the wrist cameras `cam_left_wrist` and `cam_right_wrist`; Spark_0 maps its
`cam_high` image slot to `cam_head`. `cam_third_view` remains available for video.
The three policy views retain the development calibration at 640×480:

| Camera | Focal length | Horizontal aperture | Vertical aperture |
| --- | ---: | ---: | ---: |
| Head | 10.0 | 13.899 | 10.422 |
| Left/right wrist (`Gemini_305`) | 10.0 | 17.259934 | 12.946656 |

The Spark wrapper registers the head calibration and the missing `Gemini_305`
type through its existing registration function. Existing shared camera types
keep their settings.

RGB images, instruction, env index, and robot state keep the public structure.
The policy reads `vision/<camera>/color` and packs the existing state fields:
7 EE-pose values plus 20 hand values per side, for 54 values in EE mode.
Joint mode selects the seven arm joint values per side instead of EE poses.
No whole-observation repacking is performed by the evaluator.

The remaining conversion is `extrinsic_matrix` (4×4) to `extrinsics`
(`[x, y, z, qw, qx, qy, qz]`), published alongside the original matrix. The
current Spark_0 parser accepts the latter key and shape when camera-frame EE
state/actions are enabled. Camera translation is made env-local by the existing
ObsManager flag, matching EE poses; the camera's USD axes are retained.

The shared `sim_config.yml` sets `dt=0.004`, `decimation=1` and ten environments
by default. SparkArena's `_task.yml` sets `render_interval=5` and
`data_source: teleop`; the existing configuration loader automatically enables
PhysX stabilization for teleop tasks. No separate simulation profile is needed.
The profile collects at 25 Hz, giving ten physics steps (0.04 simulated seconds)
per control/observation interval. Its `capture_render_passes=2` uses the existing
render fence before capture; render passes do not advance physics. The policy's
observation-upload/history schedule is retained. Use `EVAL_NUM_ENVS` to override
the parallel environment count for an individual run.

## Scenes, physical state and resets

Evaluation reuses the public saved-layout loader and SceneManager. The source
layouts already contain final object poses, including `retrieve_gap` relative
placements and the Tianji stand pose. The collection-time relative-placement
generator is not required to replay those layouts. The table dimensions and
height come from `tianji_tabletop.yml` and the saved layouts.

Microwave support metadata is relative to a link, rather than the object's root.
Runtime support queries use that link's current world pose, subtract the parallel
environment origin, then compose the metadata's local support transform. The
existing USD-based lookup remains available when a USD path is supplied.

Scene reset retains the existing order: restore tables, advance 20 physics steps,
restore objects, then advance another 20 steps. Rigid objects already restore
root pose and linear/angular velocity. For articulated objects marked
`reset_drive: true`, reset also clears joint velocity before restoring joint
positions and drive targets. This applies to the mouse and microwave. Authored
drive targets, angular-unit conversion, joint limits and root-velocity reset
continue to use the existing ArticulationObject implementation.

The shared evaluator already resets robots, observations, control queues and
reward progress. Spark calls the shared stability check with its existing
defaults, matching `xspark-0-simeval-server`: the first physics step supplies the
reference pose, displacement is checked independently along each axis against
0.04 m, and minimum height is the table height minus 0.05 m. The full run uses
300 steps with a 30-degree rotation limit; the final 100-step window uses
10 degrees. The existing history indexing is retained. No additional scene
manager or Spark-specific reset pipeline is added.

## Assets, dependencies and installation

Use the existing installation script from the repository root:

```bash
bash scripts/install.sh --install
conda activate RoboDojo
```

The simulator retains Python 3.11, Isaac Sim 5.1, PyTorch 2.7.0/cu128 and the
existing CuRobo/Pink installation. `git-lfs` is included in system dependencies.
Spark evaluation does not require the collection-time `dex-retargeting`,
`wuji-retargeting`, Oculus or glove dependencies.

The submodules use these recorded revisions; the XPolicyLab update adds the
Spark_0 adapter and its current Tianji/SparkArena interface:

| Submodule | Revision |
| --- | --- |
| `third_party/IsaacLab` | `afca7b09d60d8beb9c1cb28b43066499940b969b` |
| `third_party/curobo` | `d17b54ce32cba095c0b000c4c58777075d11de0e` |
| `XPolicyLab` | `ddf5dcfff5e089a323da1d22891909b72864647e` |

The installer follows recorded commits rather than updating to a moving branch.
The earlier XPolicyLab revision `53c5da5` does not contain `policy/Spark_0`.

When an existing asset checkout is available, import the evaluation resources
with the existing asset script:

```bash
bash scripts/init_assets.sh --dataset SparkArena --source /path/to/Assets
```

`--source` names the source **Assets directory**. It copies the following paths
into this checkout, preserving existing asset files and dereferencing symlinks:

| Path under `Assets/` | Purpose |
| --- | --- |
| `Robots/tianji_marvin_wuji/` | Left/right USDs, URDFs, meshes, wrist textures, descriptions and Curobo templates |
| `Object/SparkArena/` | Rigid, articulated and geometry assets, including metadata |
| `Eval_Layout/SparkArena/tianji_marvin_wuji/` | Saved evaluation layouts |
| `Room/Simple_Room_nolight/` | Room and its dependencies |
| `Material/material_0116/` and `Material/material_0122/` | Table and ground MDL/textures |
| `Background/brown_photostudio_02_4k.hdr` | Environment lighting |

The current seed-0 layout set has 55 layouts for each of the seven tasks: 50
requested episodes plus spare layouts. The source also has one seed-1
`dual_bottles_pick` layout; that is not a complete additional evaluation seed.

Spark asset setup runs `utils/update_embodiment_config_path.py` after import.
It generates Curobo paths for this repository and makes the bundled Tianji camera
texture references relative to their USD layers. Run the same command again
after moving the checkout:

```bash
python utils/update_embodiment_config_path.py
```

Run asset preparation in the simulator environment. Shared requirements include
`usd-core==26.3` for USD editing without starting Kit; existing environments can
add it with `python -m pip install usd-core==26.3`. Generated Curobo configs are
local files; source templates are preserved. Asset imports and the simulator
install do not download policy checkpoints.

Without `--source`, `init_assets.sh` keeps its Hugging Face download path and
honors `HF_REPO_ID`, `HF_REPO_URL` and `HF_REVISION`. A Spark asset mirror must
contain all paths in the table. The legacy four-directory readiness check alone
does not establish Spark readiness. If an existing Assets tree lacks Spark
resources, add them with `--source`; it is not moved aside or replaced. The local
import path has been exercised for this migration; a public Spark asset download
revision has not been verified.

Keep Spark_0's model dependencies in its separate policy environment:

```bash
conda create -n spark_0 python=3.10 -y
conda activate spark_0
bash XPolicyLab/policy/Spark_0/install.sh
conda activate RoboDojo
```

That installer owns the policy's PyTorch 2.6, FlashAttention and vendored StarVLA
dependencies. Follow `XPolicyLab/policy/Spark_0/README.md` for model weights and
checkpoint selection. Do not install those packages into the simulator
environment. Pass `--policy-env spark_0` to evaluation commands.

Check the assets and generated paths through the existing doctor command:

```bash
bash scripts/robodojo.sh doctor --dataset SparkArena \
  --env-cfg tianji_marvin_wuji --task stack_bowls \
  --policy-dir XPolicyLab/policy/Spark_0 \
  --policy-env spark_0
```

The Spark branch checks both robot descriptions/USDs/URDFs/Curobo configs, the
shared scene resources, seed-0 layouts for the selected task and the generated
URDF paths. `--skip-isaac --skip-conda --skip-policy` performs the resource/config
checks without requiring installed runtime or policy environments.

## Inspect launch arguments

A policy directory must provide its normal XPolicyLab interface files even for
the public CLI's dry run. Use an available policy checkout:

```bash
bash scripts/robodojo.sh eval \
  --dataset SparkArena \
  --env-cfg tianji_marvin_wuji \
  --policy-dir /path/to/XPolicyLab/policy/Spark_0 \
  --task dual_bottles_pick \
  --ckpt sft_spark_0 \
  --policy-env spark_0 \
  --dry-run

bash scripts/robodojo.sh smoke \
  --dataset SparkArena \
  --env-cfg tianji_marvin_wuji \
  --policy-dir /path/to/XPolicyLab/policy/Spark_0 \
  --ckpt sft_spark_0 \
  --policy-env spark_0 \
  --only dual_bottles_pick,stack_bowls \
  --dry-run
```

Sweeps select task modules with matching classes and YAMLs. Dry runs render
commands and record `DRY_RUN`; failed commands are recorded as `FAIL`. Use
`benchmark` instead of `smoke` to run each task's native episode count.

Add `--gpu-ids 0,1` to distribute a sweep across GPUs. SparkArena uses equal
task-count weights; RoboDojo retains its measured runtime weights.

## Paths and results

All paths use the selected benchmark and the profile's `config_name`:

```text
task/SparkArena/tasks/<task>.py
task/SparkArena/config/<task>.yml
Assets/Object/SparkArena/
Assets/Eval_Layout/SparkArena/<config_name>/<seed>/
eval_result/SparkArena/<task>/<policy>/<config_name>/<seed>_<run_info>/<run_id>/_result.json
```

Evaluation resume manifests share that benchmark-scoped result parent. Batch
status files and logs default to `smoke_results/SparkArena/`, and sweeps retain
the original summary format and resume behavior.

```bash
bash scripts/robodojo.sh doctor --dataset SparkArena \
  --env-cfg tianji_marvin_wuji \
  --skip-isaac --skip-conda --skip-policy

bash scripts/robodojo.sh summarize --dataset SparkArena
bash scripts/robodojo.sh summarize --dataset SparkArena \
  --ckpt sft_spark_0 --action-type ee --env-cfg tianji_marvin_wuji
```

Summary `--env-cfg` filters the **saved config_name**, rather than selecting a
profile file. `--root` overrides the default result root `eval_result/SparkArena`.
`--output` overrides
the output filename. Experiments with different profiles or run labels remain
separate. Only the latest result per experiment/task/seed is considered;
incomplete or invalid latest runs do not fall back to older completed runs.

SparkArena has no RoboDojo capability groups or standard/random pairing.
Native episode counts come from `task/SparkArena/config/_task.yml` (50 per task).
The default summary seed is 0; repeat
`--seed` to summarize other seeds explicitly. Averages include completed tasks
only and incomplete task cells remain blank.

## Infrastructure checks

```bash
bash -n scripts/robodojo.sh scripts/eval_policy.sh \
  scripts/internal/smoke_all_tasks.sh scripts/internal/verify_install.sh \
  scripts/install.sh scripts/init_assets.sh
python scripts/internal/task_inventory.py --check
python scripts/internal/task_inventory.py --dataset SparkArena --check
```
