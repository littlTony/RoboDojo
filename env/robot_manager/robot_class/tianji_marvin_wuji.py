import os

import numpy as np

from utils.pipeline_utils import get_embodiment_config_by_robot_type


class _TianjiMarvinWujiArm:
    """One separately spawned Tianji arm with a 20-joint WujiHand2."""

    side: str

    def __init__(self, cfg: dict):
        self.robot_type = cfg.get("robot_type", "arm")
        self.is_coupled = cfg.get("coupled", False)
        self.default_root_pos = cfg["default_root_pos"]
        self.default_root_rot = cfg["default_root_rot"]
        self.grasp_perfect_direction = cfg["grasp_perfect_direction"][0 if self.side == "left" else 1]
        self.ik_solver_type = cfg.get("ik_solver_type", "curobo")
        self.SceneCfg = None
        self.static_camera_list = None
        self.robot_cfg = get_embodiment_config_by_robot_type(
            robot_type=self.robot_type, robot_name="tianji_marvin_wuji", key=self.side
        )
        self.robot_file = self.robot_cfg["robot_file"]
        self.robot_args = self.robot_cfg["robot_config"]
        self.robot_name = f"{self.side}_tianji_marvin_wuji"
        self.urdf_path = os.path.join(self.robot_file, self.robot_args["urdf_path"])
        self.mesh_dir = os.path.join(self.robot_file, self.robot_args.get("mesh_dir", "urdf"))
        self.curobo_yml_path = os.path.join(self.robot_file, f"curobo_{self.side}.yml")
        self.ee_joint_name = self.robot_args["ee_joints"]
        self.ee_link_name = self.robot_args["ee_link"]
        self.base_link = self.robot_args["base_link"]
        self.arm_joints_name = list(self.robot_args["arm_joints_name"])
        self.gripper_joints_name = list(self.robot_args["gripper_joints_name"])
        self.ee_type = self.robot_args["ee_type"]
        self.entity_origin_pose = self.default_root_pos + self.default_root_rot
        identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.delta_matrix = np.array(self.robot_args.get("delta_matrix", identity))
        self.inv_delta_matrix = np.linalg.inv(self.delta_matrix)
        self.global_trans_matrix = np.array(self.robot_args.get("global_trans_matrix", identity))
        self.camera = self.robot_args.get("camera", None)


class LeftTianjiMarvinWuji(_TianjiMarvinWujiArm):
    side = "left"


class RightTianjiMarvinWuji(_TianjiMarvinWujiArm):
    side = "right"
