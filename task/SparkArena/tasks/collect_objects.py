from env.environment.task_env import TaskEnv
from env.reward_manager.reward_manager import RewardManager


class collect_objects(TaskEnv):
    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        self.step_lim = 600
        self.auto_back_home = True

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()

    def gen_instruction(self, env_idx):
        return ["Put all the objects into the basket."]

    def _object_in_basket_checks(self, label):
        return [
            self.reward_manager.is_A_in_B(label_A=label, label_B="basket"),
            self.reward_manager.is_A_depth_in_B(label_A=label, label_B="basket", z_threshold=0.0),
        ]

    def run_reward(self):
        self.reward_manager.check(
            [
                *self._object_in_basket_checks("target0"),
                *self._object_in_basket_checks("target1"),
                *self._object_in_basket_checks("target2"),
                self.reward_manager.all_robot_back_to_origin(),
            ]
        )

    def get_score(self):
        # 1 object -> 15, 2 objects -> 40, 3 objects -> 100
        self.reward_manager.score(
            [
                self._object_in_basket_checks("target0"),
                self._object_in_basket_checks("target1"),
                self._object_in_basket_checks("target2"),
            ],
            [15, 40, 100],
            score_mode="by_count",
        )
