from env.environment.task_env import TaskEnv
from env.reward_manager.reward_manager import RewardManager


class ClickMouseCommon:
    """Mouse placement and click reward logic."""

    MOUSE_LABEL = "mouse"
    MOUSEMAT_LABEL = "mousemat"
    MOUSE_BUTTON_TAG = "left_press"

    BUTTON_PRESSED_RATIO = 0.75

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        self.step_lim = 400
        self.auto_back_home = True

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()

    def _mouse_on_mousemat_checks(self):
        """Check that the mouse is on the mouse pad."""
        rm = self.reward_manager
        return [
            rm.is_A_in_B(
                label_A=self.MOUSE_LABEL,
                label_B=self.MOUSEMAT_LABEL,
            ),
        ]

    def run_reward(self):
        rm = self.reward_manager

        rm.check(
            [
                *self._mouse_on_mousemat_checks(),
                rm.is_joint_position_above_ratio(
                    label=self.MOUSE_LABEL,
                    percentage=self.BUTTON_PRESSED_RATIO,
                    tag=self.MOUSE_BUTTON_TAG,
                ),
            ]
        )
        rm.check([*self._mouse_on_mousemat_checks(), rm.all_robot_back_to_origin()])

    def gen_instruction(self, env_idx):
        return ["Place the mouse on the mouse pad, then click the left button."]


class click_mouse(ClickMouseCommon, TaskEnv):
    pass
