"""Benchmark-scoped paths shared by evaluation and scene loading."""

from dataclasses import dataclass
from pathlib import Path

from env.global_configs import ASSETS_PATH, BENCHMARK, ROOT_DIR


@dataclass(frozen=True)
class BenchmarkPaths:
    name: str = BENCHMARK

    def __post_init__(self):
        if self.name not in ("RoboDojo", "MiVerse", "SparkArena"):
            raise ValueError(f"Unknown benchmark {self.name!r}; expected 'RoboDojo', 'MiVerse', or 'SparkArena'.")

    @classmethod
    def from_config(cls, config):
        """Read an eval profile, preserving the default for existing profiles."""
        return cls(config.get("benchmark", BENCHMARK))

    @property
    def task_dir(self) -> Path:
        return Path(ROOT_DIR, "task", self.name)

    @property
    def config_dir(self) -> Path:
        return self.task_dir / "config"

    @property
    def objects_dir(self) -> Path:
        return Path(ASSETS_PATH, "Object", self.name)

    def layout_dir(self, config_name, seed) -> Path:
        return Path(ASSETS_PATH, "Eval_Layout", self.name, str(config_name), str(seed))

    def result_dir(self, eval_cfg) -> Path:
        """Parent of timestamped results and the associated resume manifest."""
        return Path(
            "eval_result",
            self.name,
            eval_cfg["task_name"],
            eval_cfg["policy_name"],
            eval_cfg["config_name"],
            f"{eval_cfg.get('seed', 0)}_{eval_cfg.get('additional_info', '')}",
        )

    def resume_manifest_path(self, eval_cfg, run_id) -> Path:
        return self.result_dir(eval_cfg) / f"_resume_{run_id}.json"
