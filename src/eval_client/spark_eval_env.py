"""SparkArena evaluation wrapper built on the normal RoboDojo EvalEnv."""

from copy import deepcopy
from functools import cache

import numpy as np

from env.benchmark import BenchmarkPaths
from src.eval_client.eval_env import (
    build_eval_env_class,
    resolve_task_class,
)
from utils.transformer import matrix_to_pose

_SPARK_HEAD_CAMERA_TYPE = "SparkArena_teleop_head"
_SPARK_TELEOP_HEAD_TEMPLATE = {
    "resolution": (640, 480),
    "focal_length": 10.0,
    "horizontal_aperture": 13.899,
    "vertical_aperture": 10.422,
    "clipping_range": (0.005, 10.0),
}
_SPARK_WRIST_CAMERA_TEMPLATE = {
    "resolution": (640, 480),
    "focal_length": 10.0,
    "horizontal_aperture": 17.259934,
    "vertical_aperture": 12.946656,
    "clipping_range": (0.005, 10.0),
}


def _with_sparkarena_observation_config(config):
    """Copy the current profile and select Spark's head-camera calibration."""

    config = deepcopy(config)
    if BenchmarkPaths.from_config(config["eval_cfg"]).name != "SparkArena":
        raise ValueError("SparkArenaEvalEnv requires eval_cfg.benchmark: SparkArena.")
    # RoboDojo's global teleop_head template has different aperture values.
    # Use a private type in the copied SparkArena config so default RoboDojo
    # camera behavior remains unchanged.
    config["camera"]["cam_head"]["camera"]["type"] = _SPARK_HEAD_CAMERA_TYPE
    return config


def _spark_extrinsic_matrix_to_pose(matrix):
    """Isaac ``extrinsic_matrix`` -> Spark training camera pose.

    ``CameraManager.get_camera_extrinsics`` is the 4x4 camera-to-env transform
    of the USD camera frame (x right, y up, z back). That is already
    ``T_world_from_camera`` in the axes Spark training records, so this only
    changes the representation to ``[x, y, z, qw, qx, qy, qz]`` with ``w >= 0``.
    The policy then reads ``vision/<camera>/extrinsics`` and keeps
    ``deploy.yml`` camera convention on ``auto``.
    """

    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(
            "SparkArena camera extrinsic_matrix must be a finite 4x4 "
            f"camera-to-env transform, got shape {transform.shape}."
        )
    pose = np.asarray(matrix_to_pose(transform), dtype=np.float64)
    if pose[3] < 0.0:
        pose[3:7] *= -1.0
    return pose


def _write_spark_camera_extrinsics(observation):
    """Publish training-convention poses next to the raw extrinsic matrices."""

    vision = observation.get("vision") if isinstance(observation, dict) else None
    if not isinstance(vision, dict):
        return
    for camera_name, camera_obs in vision.items():
        if not isinstance(camera_obs, dict) or "extrinsic_matrix" not in camera_obs:
            continue
        try:
            camera_obs["extrinsics"] = _spark_extrinsic_matrix_to_pose(camera_obs["extrinsic_matrix"])
        except ValueError as exc:
            raise ValueError(f"SparkArena camera {camera_name}: {exc}") from exc


def _register_sparkarena_camera_types():
    """Register the head calibration and the missing Wuji wrist-camera type."""

    from env.camera_manager.camera_manager import REAL_MAP

    REAL_MAP[_SPARK_HEAD_CAMERA_TYPE] = dict(_SPARK_TELEOP_HEAD_TEMPLATE)
    REAL_MAP.setdefault("Gemini_305", dict(_SPARK_WRIST_CAMERA_TEMPLATE))


@cache
def build_sparkarena_eval_env_class(task_class):
    """Build the SparkArena wrapper for a dynamically resolved task class."""

    eval_env_class = build_eval_env_class(task_class)

    class SparkArenaEvalEnv(eval_env_class):
        """SparkArena EvalEnv with Spark-compatible direct Curobo IK."""

        def __init__(self, config, app, resume_state=None, **kwargs):
            config = _with_sparkarena_observation_config(config)
            _register_sparkarena_camera_types()
            super().__init__(config, app, resume_state=resume_state, **kwargs)
            robot_manager = self.robot_manager
            self.obs_manager.extrinsics_in_env_frame = True
            # Keep the inherited EvalEnv action path while changing IK only for
            # this environment instance.
            robot_manager.solve_ik = self._solve_sparkarena_ik
            # Spark writes the authored joint state during reset. Keep this
            # behavior local to SparkArena instead of changing RobotManager.
            robot_manager.set_robot_init_pose = robot_manager.write_robot_init_joint_state_to_sim
            self._bind_sparkarena_reward_parser()

        def get_obs_batch(self, env_idx_list=None, last_frame=False):
            """Convert published camera matrices before the policy sees them."""

            observations = super().get_obs_batch(env_idx_list=env_idx_list, last_frame=last_frame)
            for observation in observations:
                _write_spark_camera_extrinsics(observation)
            return observations

        def process_control_info(self, control_info, env_idx):
            """Use legacy position control with zero arm and hand velocities.

            The shared EvalEnv interpolates arm positions for the first 80%
            of the collect interval, then holds the target. Dexterous-hand
            positions stay at the policy target throughout. The component
            controller writes arm velocities; hand velocities retain the zero
            target established by the existing initialization/reset path.
            """

            control_seq = super().process_control_info(control_info, env_idx)
            for controls in control_seq:
                for robot in self.robot_manager.robot_list:
                    for name in (robot.arm_name, robot.gripper_name):
                        key = self.robot_manager.process_name(name)
                        joint_control = controls.get(key)
                        if joint_control is not None:
                            joint_control["velocity"] = np.zeros_like(
                                joint_control["position"], dtype=np.float64
                            ).tolist()
            return control_seq

        def _bind_sparkarena_reward_parser(self):
            """Bind Spark's hand-open check only to this environment instance."""

            reward_manager = getattr(self, "reward_manager", None)
            parser = getattr(reward_manager, "func_parser", None)
            original = getattr(parser, "is_all_gripper_open", None)
            if not callable(original):
                raise RuntimeError("SparkArena requires RewardManager.func_parser.is_all_gripper_open().")
            parser.is_all_gripper_open = self._sparkarena_is_all_gripper_open

        @staticmethod
        def _sparkarena_to_numpy(value):
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            return np.asarray(value, dtype=np.float64)

        def _sparkarena_is_all_gripper_open(self, args):
            """Evaluate Spark's scalar-gripper and hand-open semantics."""

            env_idx = int(args["env_idx"])
            open_threshold = float(args["open_threshold"])
            if env_idx < 0 or env_idx >= self.num_envs:
                raise IndexError(f"SparkArena env_idx={env_idx} is out of range.")

            robot_manager = self.robot_manager
            for robot in robot_manager.robot_list:
                if getattr(robot, "type", None) != "target":
                    continue
                open_val = robot_manager.get_end_effector_real_val(robot=robot, env_idx_list=[env_idx])[env_idx]
                if robot.ee_type == "hand":
                    robot_idx = robot_manager.robot_list.index(robot)
                    articulation = robot_manager.robot_key[robot_idx]
                    joint_indices = list(robot.gripper_joint_indices)
                    default_pos = self._sparkarena_to_numpy(
                        articulation.data.default_joint_pos[env_idx][joint_indices]
                    ).reshape(-1)
                    joint_limits = self._sparkarena_to_numpy(
                        articulation.data.soft_joint_pos_limits[env_idx, joint_indices]
                    )
                    current_pos = self._sparkarena_to_numpy(open_val).reshape(-1)
                    joint_range = np.maximum(joint_limits[:, 1] - joint_limits[:, 0], 1e-6)
                    open_ratio = 1.0 - np.mean(
                        np.clip(
                            np.abs(current_pos - default_pos) / joint_range,
                            0.0,
                            1.0,
                        )
                    )
                elif robot.ee_type == "gripper":
                    open_array = self._sparkarena_to_numpy(open_val).reshape(-1)
                    scale = self._sparkarena_to_numpy(robot.gripper_scale).reshape(-1)
                    scale_range = scale[1] - scale[0]
                    open_ratio = (np.mean(open_array) - scale[0]) / scale_range
                else:
                    raise RuntimeError(f"Unsupported SparkArena target ee_type={robot.ee_type!r}.")
                if open_ratio < open_threshold:
                    return 0.0
            return 1.0

        def _solve_sparkarena_ik(
            self,
            target_pose,
            env_idx,
            robot,
            trans="world",
        ):
            """Run Spark's direct Curobo IK without changing shared managers."""

            if trans != "world":
                raise ValueError("SparkArena direct IK only supports world-frame EE poses.")

            target_pose = np.asarray(target_pose, dtype=np.float64)
            if target_pose.shape != (7,) or not np.all(np.isfinite(target_pose)):
                raise ValueError(f"SparkArena EE target must be finite with shape (7,), got {target_pose.shape}.")

            from env.planner_manager.curobo_planner import CuroboPlanner

            # The public manager stores CuroboPlanner directly, rather than
            # the development repo's PlannerInterface/ik_implementation pair.
            planner = self.robot_manager.get_planner(robot)
            if not isinstance(planner, CuroboPlanner):
                raise RuntimeError("SparkArena requires the Curobo IK backend.")
            planner_joints = list(planner.active_joints_name)
            robot_arm_joints = list(getattr(robot, "arm_joints_name", []))
            if planner_joints != robot_arm_joints or not planner_joints:
                raise RuntimeError(
                    "SparkArena requires the planner joint order to match the "
                    "robot embodiment; "
                    f"planner={planner_joints}, robot={robot_arm_joints}."
                )

            current_qpos = self.robot_manager.get_named_joint_positions(
                planner_joints,
                int(env_idx),
                robot=robot,
            )
            # Keep the same env-local base-pose precedence as the development
            # manager, using fields already provided by the public manager.
            real_robot_pose = getattr(robot, "curobo_base_pose", None)
            if real_robot_pose is None:
                real_robot_pose = getattr(robot, "base_link_origin_pose", None)
            if real_robot_pose is None:
                real_robot_pose = robot.entity_origin_pose
            result = planner.solve_ik_to_joint(
                current_qpos,
                target_pose,
                real_robot_pose=deepcopy(real_robot_pose),
            )
            if not isinstance(result, dict):
                raise RuntimeError("Curobo solve_ik_to_joint() returned a non-dict result.")
            if result.get("status") != "Success":
                return result

            result_names = list(result.get("joint_names") or [])
            if result_names != planner_joints:
                raise RuntimeError(
                    "Curobo IK joint order does not match the arm planner; "
                    f"result={result_names}, planner={planner_joints}."
                )
            joint_value = np.asarray(result.get("joint_value"), dtype=np.float64)
            expected_shape = (len(planner_joints),)
            if joint_value.shape != expected_shape or not np.all(np.isfinite(joint_value)):
                raise RuntimeError(
                    "Curobo IK must return finite joint_value matching the "
                    f"planner dimension {expected_shape}, got {joint_value.shape}."
                )
            result["joint_value"] = joint_value
            return result

    return SparkArenaEvalEnv


def create_sparkarena_eval_env(config, app, resume_state=None, **kwargs):
    """Create a SparkArena EvalEnv for the task selected by ``config``."""

    env_class = build_sparkarena_eval_env_class(resolve_task_class(config))
    return env_class(config, app, resume_state=resume_state, **kwargs)


__all__ = [
    "build_sparkarena_eval_env_class",
    "create_sparkarena_eval_env",
]
