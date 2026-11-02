from env.environment.task_env import TaskEnv
from env.reward_manager.reward_manager import RewardManager


class hammer_beat(TaskEnv):
    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        self.step_lim = 300

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()

    def gen_instruction(self, env_idx):
        return ["Pick up the hammer and beat the cube with the hammer head."]

    def run_reward(self):
        self.reward_manager.check(
            [
                self.reward_manager.is_lift(label="hammer", z_threshold=0.02),
                self.reward_manager.is_AB_xy_distance_within_threshold(
                    label_A="hammer",
                    label_B="cube",
                    A_functional_point="head",
                    A_point_type="active",
                    threshold=0.025,
                ),
                [
                    self.reward_manager.is_axis_up(label="hammer", axis=[1, 0, 0], threshold=45),
                    self.reward_manager.is_axis_up(label="hammer", axis=[-1, 0, 0], threshold=45),
                ],
                self.reward_manager.is_A_point_above_B_point_by_z_range(
                    label_A="hammer",
                    label_B="cube",
                    A_functional_tag="head",
                    A_type="active",
                    B_support_point="block/0",
                    z_upper=0.04,
                    z_lower=-0.01,
                ),
            ]
        )
