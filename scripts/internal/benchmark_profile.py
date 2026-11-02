#!/usr/bin/env python3
"""Resolve benchmark defaults and eval profiles without simulation imports."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

# Only benchmark-specific defaults belong here; execution stays shared.
BENCHMARKS = {
    "MobileArena": {
        "default_env_cfg": "mobile_arena",
        "default_task": "place_bread_in_oven",
        "summary_script": "scripts/MobileArena/summarize_result.py",
    },
    "SparkArena": {
        "default_env_cfg": "tianji_marvin_wuji",
        "default_task": "stack_bowls",
        "summary_script": "scripts/SparkArena/summarize_result.py",
    },
}


def expected_episode_count(bench: str, task: str, requested: str = "native") -> int:
    """Match main.py's task default and cap for an explicit episode request."""
    import yaml

    if bench not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {bench!r}")
    path = ROOT_DIR / "task" / bench / "config" / "_task.yml"
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    common = spec.get("common", {}).get("eval_nums", 50)
    native = (spec.get("tasks", {}).get(task) or {}).get("eval_nums", common)
    if type(native) is not int or native < 1:
        raise ValueError(f"Invalid native eval count for {bench}/{task}: {native!r}")
    if requested == "native":
        return native
    if not str(requested).isdigit() or int(requested) < 1:
        raise ValueError("Episode count must be a positive integer or native")
    return min(int(requested), native)


def resolve_profile(bench: str = "MobileArena", env_cfg: str = "") -> tuple[str, str, str]:
    if bench not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {bench!r}; choose from {', '.join(BENCHMARKS)}.")
    try:
        import yaml
    except ImportError as exc:
        raise ValueError(
            "PyYAML is required to read env_cfg profiles; use the installed simulator environment."
        ) from exc

    env_cfg = env_cfg or BENCHMARKS[bench]["default_env_cfg"]
    path = ROOT_DIR / "env_cfg" / f"{env_cfg}.yml"
    try:
        profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid eval profile {path}: {exc}") from exc
    if not isinstance(profile, dict):
        raise ValueError(f"Invalid eval profile: {path}")
    profile_bench = profile.get("benchmark", "MobileArena")
    if profile_bench != bench:
        raise ValueError(f"--env-cfg {env_cfg} uses benchmark {profile_bench!r}, not {bench!r}.")
    config_name = profile.get("config_name")
    if not isinstance(config_name, str) or not config_name.strip():
        raise ValueError(f"Missing config_name in {path}")
    return bench, env_cfg, config_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", "--dataset", choices=tuple(BENCHMARKS), default="MobileArena")
    parser.add_argument("--env-cfg", default="", help="env_cfg stem; defaults to the selected benchmark's profile.")
    parser.add_argument(
        "--field",
        choices=("default_env_cfg", "default_task", "summary_script"),
        help="Print a benchmark default without reading YAML.",
    )
    args = parser.parse_args()
    if args.field:
        print(BENCHMARKS[args.bench][args.field])
        return
    try:
        resolved = resolve_profile(args.bench, args.env_cfg)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print("\t".join(resolved))


if __name__ == "__main__":
    main()
