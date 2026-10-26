from env.environment.task_env import TaskEnv
from env.reward_manager.reward_manager import RewardManager


class put_food_in_microwave(TaskEnv):
    BREAD_LABEL = "bread"
    MICROWAVE_LABEL = "microwave"
    MICROWAVE_INTERIOR_TAG = "microwave/0"
    MICROWAVE_DOOR_TAG = "door"

    BREAD_Z_LOWER = 0.015
    BREAD_Z_UPPER = 0.18
    DOOR_CLOSED_RATIO = 0.9

    def __init__(self, config, app, **kwargs):
        # This asset uses overlapping OmniGlass panes in the microwave door.
        # RTX translucency can make opaque robot links behind/intersecting those
        # panes appear transparent, so disable it only for this task.
        config.sim.setdefault("render", {})
        config.sim.render.enable_translucency = False
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        self.step_lim = 450
        self.auto_back_home = True

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()

    def gen_instruction(self, env_idx):
        return ["Put the sandwich in the microwave, then close the door."]

    def _bread_in_microwave_checks(self):
        rm = self.reward_manager
        return [
            rm.is_A_in_B_support_circle(
                label_A=self.BREAD_LABEL,
                label_B=self.MICROWAVE_LABEL,
                B_support_tag=self.MICROWAVE_INTERIOR_TAG,
            ),
            rm.is_A_point_above_B_point_by_z_range(
                label_A=self.BREAD_LABEL,
                label_B=self.MICROWAVE_LABEL,
                B_support_point=self.MICROWAVE_INTERIOR_TAG,
                z_lower=self.BREAD_Z_LOWER,
                z_upper=self.BREAD_Z_UPPER,
            ),
        ]

    def _door_closed_check(self):
        return self.reward_manager.is_joint_position_above_ratio(
            label=self.MICROWAVE_LABEL,
            percentage=self.DOOR_CLOSED_RATIO,
            tag=self.MICROWAVE_DOOR_TAG,
        )

    def run_reward(self):
        rm = self.reward_manager

        rm.check(
            [
                *self._bread_in_microwave_checks(),
                self._door_closed_check(),
                rm.all_robot_back_to_origin(),
            ]
        )
