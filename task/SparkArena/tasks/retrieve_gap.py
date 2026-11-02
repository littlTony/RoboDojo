from env.environment.task_env import TaskEnv
from env.reward_manager.reward_manager import RewardManager


class retrieve_gap(TaskEnv):
    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        self.step_lim = 800
        self.auto_back_home = True

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()

    def gen_instruction(self, env_idx):
        return ["Move the two cubes aside upright, then place the garage on the cushion."]

    def _cube_checks(self):
        checks = []
        for label in ("cube_left", "cube_right"):
            checks.extend(
                [
                    self.reward_manager.is_moved(label=label, dis_threshold=0.01),
                    self.reward_manager.is_axis_up(label=label, axis=[0, 0, 1], threshold=15),
                ]
            )
        return checks

    def _garage_checks(self):
        return [
            self.reward_manager.is_A_up_B(
                label_A="garage",
                label_B="cube_cushion",
                z_threshold_min=0.005,
                z_threshold_max=0.15,
            ),
            self.reward_manager.is_axis_up(label="garage", axis=[0, 0, 1]),
            self.reward_manager.is_AB_xy_distance_within_threshold(
                label_A="garage",
                label_B="cube_cushion",
                threshold=0.05,
            ),
        ]

    def run_reward(self):
        self.reward_manager.check(
            [
                *self._cube_checks(),
                *self._garage_checks(),
                self.reward_manager.all_robot_back_to_origin(),
            ]
        )
