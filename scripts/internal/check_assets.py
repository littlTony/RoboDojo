#!/usr/bin/env python3
"""Check selected task asset files without importing the simulator or downloading."""

import argparse
import json
from pathlib import Path
import re
import sys

from assets import LFS_HEADER
from benchmark_profile import expected_episode_count
import yaml

OBJECT_TYPES = ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid")
ROBOT_USDS = {"m92uw": "M92UW.usd"}


class AssetCheck:
    def __init__(self, root, bench):
        self.root = root
        self.assets = root / "Assets"
        self.objects = self.assets / "Object" / bench
        self.errors = []
        self.checked = {}

    def file(self, path):
        path = Path(path)
        if path in self.checked:
            return self.checked[path]
        try:
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing or empty asset file: {path}")
            with path.open("rb") as stream:
                if stream.read(len(LFS_HEADER)) == LFS_HEADER:
                    raise ValueError(f"Unresolved Git LFS pointer: {path}")
        except (OSError, ValueError) as exc:
            self.errors.append(str(exc))
            self.checked[path] = False
        else:
            self.checked[path] = True
        return self.checked[path]

    def mapping(self, path, *, json_file=False, expanded_paths=False):
        if not self.file(path):
            return {}
        try:
            text = Path(path).read_text(encoding="utf-8")
            value = json.loads(text) if json_file else yaml.safe_load(text)
            if not isinstance(value, dict):
                raise ValueError("expected a mapping")
            if expanded_paths and ("${ASSETS_PATH}" in text or "$ASSETS_PATH" in text):
                raise ValueError(
                    "unexpanded ASSETS_PATH; run utils/update_embodiment_config_path.py from the repo root"
                )
            return value
        except (OSError, ValueError, yaml.YAMLError) as exc:
            self.errors.append(f"Invalid asset/config {path}: {exc}")
            return {}

    def package(self, directory, pattern):
        paths = sorted(directory.glob(pattern))
        if not paths:
            self.errors.append(f"Missing asset files: {directory}/{pattern}")
        for path in paths:
            self.file(path)

    def reference(self, base, value, label):
        if not isinstance(value, str) or not value.strip():
            self.errors.append(f"Missing {label} in {base}/robot_config.yml")
            return
        self.file(base / value)

    def robots(self, config):
        robots = config.get("robots", [])
        if not robots:
            self.errors.append("Selected robot configuration contains no robots")
        for robot in robots:
            name = robot["robot_name"]
            if name in ("tianji_marvin_wuji_left", "tianji_marvin_wuji_right"):
                self.tianji_robot(robot)
                continue
            base = self.assets / "Robots" / name
            if name not in ROBOT_USDS:
                self.errors.append(f"Unsupported robot asset preflight: {name}")
                continue
            self.file(base / ROBOT_USDS[name])
            description = self.mapping(base / "robot_config.yml")
            if not description:
                self.errors.append(f"Missing or empty robot configuration: {base}/robot_config.yml")
                continue
            common = description.get("common", {})
            self.reference(base, common.get("urdf_path"), "urdf_path")
            if not robot.get("need_planner", True):
                continue
            planner = {**(description.get("planner") or {}), **(robot.get("planner") or {})}
            planner_path = planner.get("config") or common.get("curobo_yml")
            if not planner_path:
                self.errors.append(f"Missing planner config reference in {base}/robot_config.yml")
            else:
                if not self.mapping(base / planner_path, expanded_paths=True):
                    self.errors.append(f"Missing or empty planner configuration: {base / planner_path}")

    def tianji_robot(self, robot):
        side = robot["robot_name"].rsplit("_", 1)[-1]
        base = self.assets / "Robots/tianji_marvin_wuji"
        self.file(base / f"tianji_marvin_wuji_{side}.usd")
        description = self.mapping(base / f"{side}_robot_config.yml")
        if not description:
            self.errors.append(f"Missing or empty Tianji {side} robot description")
            return
        self.reference(base, description.get("urdf_path"), "urdf_path")
        if not robot.get("need_planner", True):
            return
        config = self.mapping(base / f"curobo_{side}.yml", expanded_paths=True)
        urdf = Path(config.get("robot_cfg", {}).get("kinematics", {}).get("urdf_path", ""))
        expected = base / f"urdf/tianji_wujihand2_{side}.urdf"
        if not urdf.is_absolute() or urdf.resolve() != expected.resolve():
            self.errors.append(f"Stale Tianji {side} Curobo URDF path; run utils/update_embodiment_config_path.py")
        self.file(urdf)

    def scene(self, config):
        room = config.get("Room")
        if room:
            self.package(self.assets / "Room" / room.get("default", "base0"), "*.usd")
        background = config.get("Background")
        if background:
            name = background.get("category_name") or background.get("default", "brown_photostudio_02_16k.hdr")
            self.file(self.assets / "Background" / name)
        table = config.get("Table")
        if table:
            self.package(self.assets / "Material" / table.get("default", "material_0001"), "*.mdl")
        ground = config.get("Ground") or {}
        material = ground.get("materials") or {}
        if ground.get("spawn", True) and material.get("type", "color") == "mdl":
            name = material.get("default")
            if name:
                path = self.assets / "Material" / name
                if path.suffix == ".mdl":
                    self.file(path)
                else:
                    self.package(path, "**/*.mdl")
            elif material.get("random"):
                self.package(self.assets / "Material", "material_*/**/*.mdl")
            else:
                self.errors.append("Ground MDL material has no default asset")

    def object(self, kind, category, index, *, cluttered=False):
        base = self.objects / ("Clutter" if cluttered else kind) / category / f"{index:05d}"
        usd = base / "object.usdz"
        if not usd.exists():
            # Match the replay loader's USD fallback, including clutter records.
            usd = self.objects / kind / category / f"{index:05d}" / "object.usd"
        self.file(usd)
        metadata = self.mapping(base / "metadata.json", json_file=True)
        if not metadata.get("geometry"):
            self.errors.append(f"Missing geometry metadata: {base}/metadata.json")

    def layout_objects(self, layout):
        found = False
        for kind in OBJECT_TYPES:
            for category, instances in layout.get(kind, {}).items():
                for instance in instances:
                    found = True
                    self.object(
                        kind, category, int(instance["category_idx"]), cluttered=instance.get("type") == "cluttered"
                    )
        return found

    def declared_objects(self, config):
        for kind in OBJECT_TYPES:
            for group in config.get(kind, []):
                for category in group.get("category", []):
                    name = category["name"]
                    indices = category.get("index")
                    if indices is None:
                        directory = self.objects / kind / name
                        indices = (
                            [int(path.name) for path in directory.iterdir() if path.is_dir() and path.name.isdigit()]
                            if directory.is_dir()
                            else []
                        )
                        if not indices:
                            self.errors.append(f"Missing object asset models: {directory}")
                    for index in indices:
                        self.object(kind, name, int(index))


def check_assets(root, bench, env_cfg, task, seed, eval_num):
    check = AssetCheck(root, bench)
    profile = check.mapping(root / "env_cfg" / f"{env_cfg}.yml")
    index = check.mapping(root / "task" / bench / "config" / "_task.yml")
    settings = {**index.get("common", {}), **index.get("tasks", {}).get(task, {})}
    refs = dict(profile.get("config", {}))
    # Keep the same override behavior as utils.pipeline_utils.process_config.
    for kind in ("scene", "robot"):
        override = settings.get(f"{kind}_config")
        if override is not None:
            refs[kind] = override
    robot = check.mapping(root / "env_cfg" / "robot" / f"{refs.get('robot')}.yml")
    scene = check.mapping(root / "env_cfg" / "scene" / f"{refs.get('scene')}.yml")
    task_cfg = check.mapping(root / "task" / bench / "config" / f"{task}.yml")
    check.robots(robot)
    layouts = root / "Assets" / "Eval_Layout" / bench / profile.get("config_name", env_cfg) / str(seed)
    pattern = re.compile(rf"{re.escape(task)}_\d+\.json")
    paths = sorted(
        (path for path in layouts.glob("*.json") if pattern.fullmatch(path.name)),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    required = expected_episode_count(bench, task, eval_num)
    selected = [check.mapping(path, json_file=True) for path in paths[:required]]
    for layout in selected or [{}]:
        check.scene({**scene, **layout})
        if not check.layout_objects(layout):
            check.declared_objects(scene)
            check.declared_objects(task_cfg)
    if check.errors:
        raise ValueError("\n".join(dict.fromkeys(check.errors)))
    return (
        f"{bench}/{task}: {len(check.checked)} asset/config files checked "
        f"(robot={refs['robot']}, scene={refs['scene']})"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--env-cfg", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-num", default="native")
    args = parser.parse_args()
    try:
        print(check_assets(args.root, args.bench, args.env_cfg, args.task, args.seed, args.eval_num))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
