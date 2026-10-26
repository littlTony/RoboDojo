#!/usr/bin/env python3
"""Summarize SparkArena results without merging checkpoints, profiles or seeds."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from env.benchmark import BenchmarkPaths


def task_spec() -> tuple[dict[str, int], list[int]]:
    import yaml

    paths = BenchmarkPaths("SparkArena")
    config = yaml.safe_load((paths.config_dir / "_task.yml").read_text(encoding="utf-8"))
    default = config["common"]["eval_nums"]
    overrides = config.get("tasks", {})
    names = sorted(path.stem for path in (paths.task_dir / "tasks").glob("*.py") if path.name != "__init__.py")
    counts = {name: (overrides.get(name) or {}).get("eval_nums", default) for name in names}
    if any(type(count) is not int or count < 1 for count in counts.values()):
        raise ValueError("SparkArena native episode counts must be positive integers")
    return counts, [0]


def load_entries(path: Path) -> list[tuple[bool, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("details"), dict):
        raise ValueError("expected an object containing details")
    entries = {}
    for key, value in data["details"].items():
        if not re.fullmatch(r"\d+", key) or not isinstance(value, dict) or int(key) in entries:
            raise ValueError(f"invalid or duplicate episode index {key!r}")
        success, score = value.get("success"), value.get("score")
        if type(success) is not bool:
            raise ValueError(f"episode {key}: success must be a boolean")
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"episode {key}: score must be finite and in [0, 1]")
        entries[int(key)] = (success, float(score))
    return [entries[index] for index in sorted(entries)]


def cell(value) -> str:
    return html.escape(str(value)).replace("|", "&#124;").replace("\n", " ")


def summarize(root: Path, output: Path, counts: dict[str, int], seeds: list[int], args) -> None:
    if not root.is_dir():
        raise ValueError(f"Eval result directory not found: {root}")
    results = {}
    experiments = set()
    for task in counts:
        for run_dir in sorted((root / task).glob("*/*/*")):
            if not run_dir.is_dir():
                continue
            policy, profile, run_label = run_dir.relative_to(root / task).parts
            match = re.fullmatch(r"(\d+)_(.*)", run_label)
            if match is None or int(match[1]) not in seeds:
                continue
            seed, run_info = int(match[1]), match[2]
            metadata = dict(item.split("=", 1) for item in run_info.split(",") if "=" in item)
            if args.env_cfg and profile != args.env_cfg:
                continue
            if args.ckpt and metadata.get("ckpt_name") != args.ckpt:
                continue
            if args.action_type and metadata.get("action_type") != args.action_type:
                continue
            experiment = (policy, profile, run_info)
            experiments.add(experiment)
            files = sorted(run_dir.glob("*/_result.json"))
            if not files:
                continue
            latest = files[-1]
            try:
                results[experiment, seed, task] = load_entries(latest)
            except (OSError, ValueError) as exc:
                # Invalid/latest partial runs never fall back to an older result.
                print(f"Warning: ignoring {latest}: {exc}", file=sys.stderr)

    lines = [
        "# SparkArena Evaluation Summary",
        "",
        f"Native episode counts are required for a complete task; seeds: {', '.join(map(str, seeds))}.",
        "Means include completed tasks only. Incomplete cells remain blank.",
        "",
        "| Experiment | Seed | Complete tasks | Mean SR (%) | Mean score (%) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    detail_lines = []
    for experiment in sorted(experiments):
        label = cell(" / ".join(experiment))
        detail_lines += [
            f"## {label}",
            "",
            "| Task | Seed | Episodes | SR (%) | Score (%) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for seed in seeds:
            completed = []
            for task, required in counts.items():
                entries = results.get((experiment, seed, task), [])
                sr, score = "", ""
                if len(entries) >= required:
                    selected = entries[:required]
                    metrics = (
                        sum(success for success, _ in selected) / required * 100,
                        sum(value for _, value in selected) / required * 100,
                    )
                    completed.append(metrics)
                    sr, score = (f"{value:.2f}" for value in metrics)
                detail_lines.append(f"| {cell(task)} | {seed} | {len(entries)}/{required} | {sr} | {score} |")
            means = [f"{statistics.mean(values):.2f}" for values in zip(*completed)] if completed else ["", ""]
            lines.append(f"| {label} | {seed} | {len(completed)}/{len(counts)} | {means[0]} | {means[1]} |")
        detail_lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join([*lines, "", *detail_lines]) + "\n", encoding="utf-8")
    print(f"SparkArena summary: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DIR / "eval_result/SparkArena")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, action="append", help="Repeat to select seeds; defaults to 0.")
    parser.add_argument("--ckpt")
    parser.add_argument("--action-type")
    parser.add_argument("--env-cfg", help="Filter by the saved config_name directory.")
    args = parser.parse_args()
    try:
        counts, seeds = task_spec()
        if args.seed is not None and (any(seed < 0 for seed in args.seed) or len(set(args.seed)) != len(args.seed)):
            raise ValueError("--seed values must be unique nonnegative integers")
        summarize(args.root, args.output or args.root / "_summary.md", counts, args.seed or seeds, args)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
