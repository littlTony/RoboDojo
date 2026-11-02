from env.environment.task_env import TaskEnv
from env.reward_manager.reward_manager import RewardManager


class dual_bottles_pick(TaskEnv):
    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        self.step_lim = 400

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()

    def gen_instruction(self, env_idx):
        return ["Pick up both bottles on the table."]

    def run_reward(self):
        self.reward_manager.check(
            [
                self.reward_manager.is_lift(label="bottle0", z_threshold=0.2),
                self.reward_manager.is_lift(label="bottle1", z_threshold=0.2),
            ]
        )

    def get_score(self):
        # 1 bottle lifted -> 30, both bottles -> 100
        self.reward_manager.score(
            [
                [self.reward_manager.is_lift(label="bottle0", z_threshold=0.2)],
                [self.reward_manager.is_lift(label="bottle1", z_threshold=0.2)],
            ],
            [30, 100],
            score_mode="by_count",
        )
