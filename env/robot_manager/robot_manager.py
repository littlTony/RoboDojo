from collections.abc import Sequence
from copy import deepcopy
from importlib import import_module
import os
from typing import List, Literal

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import transforms3d as t3d

from env.environment.isaac.isaac_rl_env import IsaacRLEnv
from env.environment.isaac.physx_device import as_numpy, ensure_host_joint_targets
from env.global_configs import ENV_REGEX_NAMESPACE
from env.robot_manager.collision_bbox import parse_configured_collision_bbox
from env.robot_manager.control_manager import ControlManager, MetaControl
from utils.ensure_usd_path import ensure_usd_path
from utils.transformer import matrix_to_pose, pose_to_matrix


def nearest_body_frame(urdf, control_frame, articulation_body_names):
    """Resolve a tool frame merged into an ancestor rigid body in the USD."""
    body_names = set(articulation_body_names)
    child_to_parent = {joint.child: joint.parent for joint in urdf.joint_map.values()}
    current = control_frame
    visited = set()
    while current not in body_names:
        if current in visited:
            raise ValueError(f"Cycle while resolving control frame {control_frame!r}.")
        visited.add(current)
        if current not in child_to_parent:
            raise KeyError(f"No articulation-body ancestor found for {control_frame!r}.")
        current = child_to_parent[current]
    return current, np.asarray(urdf.get_transform(control_frame, current), dtype=np.float64)


class RobotManager:
    def __init__(
        self,
        num_envs: int,
        env_spacing: int,
        config: DictConfig,
        dt,
        device,
    ):
        self.sim: IsaacRLEnv = None
        self.control_manager = ControlManager(num_envs=num_envs, robot_manager=self)
        self.num_envs = num_envs
        self.env_spacing = env_spacing
        self.robots_cfg = config.get("robots", [])
        self._configured_collision_bboxes = [
            parse_configured_collision_bbox(
                cfg.get("collision_bbox"), config_path=f"robot.robots[{index}].collision_bbox"
            )
            for index, cfg in enumerate(self.robots_cfg)
        ]
        self._pink_overrides = dict(config.get("pink", {}) or {})
        self.device = device
        self.dt = dt
        self.mobile_config = config.get("mobile", {}) or {}
        self.mobile_base_adapter = None
        self._last_applied_root_velocities = {}
        self._head_pose_cache = None
        self._host_state = None
        self._seeds_per_env: list[int] | None = None

        self.robot_list = []
        self.robot_key = []
        self.target_arm_nums = 0
        self.use_scene_cfg = []
        self.planner = dict()
        self.ik_solver = dict()
        self._planner_setup_requests = []
        self.robot_origin_endpose = None
        self.robot_init_joint = [dict() for _ in range(self.num_envs)]
        for idx, cfg in enumerate(self.robots_cfg):
            if not cfg.get("coupled", False):
                robot = self._get_robot(cfg, idx)
                self.robot_list.append(robot)
                self.use_scene_cfg.append(True)
                if robot.robot_name not in self.planner and cfg.get("need_planner", True):
                    self._setup_planner(robot)
                robot.type = cfg.get("type", "target")
                self.target_arm_nums += 1 if robot.type == "target" else 0
            else:
                left_robot, right_robot = self._get_robot(cfg, idx)
                self.robot_list.append(left_robot)
                self.use_scene_cfg.append(True)
                self.robot_list.append(right_robot)
                self.use_scene_cfg.append(False)
                if cfg.get("need_planner", True):
                    self._planner_setup_requests.append((left_robot, right_robot))
                left_robot.type = cfg.get("type", "target")
                right_robot.type = cfg.get("type", "target")
                self.target_arm_nums += 1 if left_robot.type == "target" else 0
                self.target_arm_nums += 1 if right_robot.type == "target" else 0

        self._planner_setup_specs = list(self._planner_setup_requests)
        self._set_robot_obs_name()

    @staticmethod
    def _apply_room_robot_pose(config, room_name):
        """Apply the selected MiVerse room's authored pose before constructing robots."""
        navigation = config.task_env.get("Navigation", {})
        if not navigation:
            return None
        if room_name not in navigation:
            raise ValueError(f"Navigation has no robot pose for room {room_name!r}.")
        pose = navigation[room_name].get("robot_pose")
        if pose is None:
            return None
        position = np.asarray(pose.get("position"), dtype=np.float64)
        orientation = np.asarray(pose.get("orientation"), dtype=np.float64)
        if position.shape != (3,) or orientation.shape != (4,):
            raise ValueError("Navigation.robot_pose requires xyz position and wxyz orientation.")
        robots = config.robot.get("robots", [])
        if not robots:
            raise ValueError("Navigation.robot_pose is configured but robot.robots is empty.")
        for robot in robots:
            robot["default_root_pos"] = position.tolist()
            robot["default_root_rot"] = orientation.tolist()
        return np.concatenate((position, orientation))

    def initialize_planners(self):
        """Build coupled IK after scene initialization, once per scene lifetime."""
        for robots in self._planner_setup_requests:
            self._setup_tool_ik(robots)
        self._planner_setup_requests.clear()

    def update_env_seeds(self, seeds: Sequence[int] | None):
        if seeds is None:
            self._seeds_per_env = None
            return
        seed_list = [int(s) for s in seeds]
        if len(seed_list) != self.num_envs:
            raise ValueError(f"seed list length {len(seed_list)} does not match num_envs {self.num_envs}.")
        self._seeds_per_env = seed_list

    def get_robot_obs_name(self):
        obs_list = []
        for idx, robot in enumerate(self.robot_list):
            obs_list.append(self.process_name(robot.arm_name))
            obs_list.append(self.process_name(robot.gripper_name))
        return obs_list

    def set_robot_init_pose(self):
        seen = set()
        for idx, robot in enumerate(self.robot_list):
            key = self.robot_key[idx]
            if id(key) in seen:
                continue
            seen.add(id(key))
            if robot.robot_type == "arm":
                target_joints = key.data.default_joint_pos.clone()
                target_vel = key.data.default_joint_vel.clone()
                key.set_joint_position_target(target_joints)
                key.set_joint_velocity_target(target_vel)

    def write_robot_init_joint_state_to_sim(self, env_ids=None):
        """Re-pin Eval joints to the authored pose after scene settling."""
        if not self.robot_key:
            self._setup_robot_key()
        indices = list(range(self.num_envs)) if env_ids is None else [int(index) for index in env_ids]
        if not indices:
            return
        if any(index < 0 or index >= self.num_envs for index in indices):
            raise IndexError(f"Robot init-state env indices are out of range: {indices}.")
        self._host_state = None
        seen = set()
        for articulation in self.robot_key:
            if id(articulation) in seen:
                continue
            seen.add(id(articulation))
            sim_env_ids = torch.as_tensor(indices, device=articulation.device, dtype=torch.int32)
            row_ids = sim_env_ids.to(dtype=torch.long)
            positions = articulation.data.default_joint_pos.index_select(0, row_ids).clone()
            velocities = torch.zeros_like(positions)
            articulation.write_joint_state_to_sim(positions, velocities, env_ids=sim_env_ids)
            articulation.set_joint_position_target(positions, env_ids=sim_env_ids)
            articulation.set_joint_velocity_target(velocities, env_ids=sim_env_ids)
        if self.mobile_base_adapter is not None:
            self.mobile_base_adapter.reset(indices)
        for env_idx in indices:
            self._last_applied_root_velocities.pop(env_idx, None)

    def set_robot_init_state(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = range(self.num_envs)

        state_dict = {env_idx: {} for env_idx in env_idx_list}
        for idx, robot in enumerate(self.robot_list):
            if robot.robot_type == "arm":
                joint_list = self.get_joint(robot=robot)
                if robot.ee_type == "gripper":
                    gripper_scale = robot.gripper_scale
                    val = gripper_scale[1] if robot.gripper_move["sign"] == 1 else gripper_scale[0]
                    mimic = robot.gripper_move["mimic"]
                    gripper_list = [[val, val * mimic[1] + mimic[2]] for _ in range(len(joint_list))]
                elif robot.ee_type == "hand":
                    gripper_list = self.get_end_effector_real_val(robot=robot)
                for env_idx in env_idx_list:
                    state_dict[env_idx][self.process_name(robot.arm_name)] = {
                        "position": joint_list[env_idx],
                        "velocity": [0.0] * len(joint_list[env_idx]),
                    }
                    state_dict[env_idx][self.process_name(robot.gripper_name)] = {
                        "position": gripper_list[env_idx],
                        "velocity": [0.0] * len(gripper_list[env_idx]),
                    }

        for env_idx in env_idx_list:
            self.control_manager.update_prev_control(env_idx, MetaControl(state_dict[env_idx]))

    def set_origin_endpose(self):
        for idx, robot in enumerate(self.robot_list):
            delta_endpose = self.get_delta_endpose(robot)
            for env_idx in range(self.num_envs):
                self.robot_origin_endpose[env_idx][robot.arm_name] = deepcopy(delta_endpose[env_idx])

    def refresh_host_state(self):
        """Share one CPU snapshot per physical robot across control and pose reads."""
        host = {}
        for articulation in self.robot_key:
            key = id(articulation)
            if key in host:
                continue
            entry = {}
            for attr in ("joint_pos", "root_pose_w", "body_link_pose_w"):
                value = getattr(articulation.data, attr, None)
                if value is not None:
                    entry[attr] = np.array(as_numpy(value), copy=True)
            host[key] = entry
        origins = getattr(self.scene, "env_origins", None)
        if origins is not None:
            host["env_origins"] = np.array(as_numpy(origins), copy=True)
        self._host_state = host

    def _host_array(self, articulation, attr):
        cache = getattr(self, "_host_state", None)
        entry = None if cache is None else cache.get(id(articulation))
        if entry is not None and attr in entry:
            return entry[attr]
        return as_numpy(getattr(articulation.data, attr))

    def _host_env_origins(self):
        cache = getattr(self, "_host_state", None)
        if cache is not None and "env_origins" in cache:
            return cache["env_origins"]
        return as_numpy(self.scene.env_origins)

    def get_real_endpose(self, robot, env_idx_list=None, is_relative=True):
        link_name = robot.ee_link_name
        return self.get_link_pose(
            robot,
            link_name=link_name,
            env_idx_list=env_idx_list,
            is_relative=is_relative,
        )

    def get_link_pose(self, robot, link_name, env_idx_list=None, is_relative=False):
        if env_idx_list is None:
            env_idx_list = list(range(self.num_envs))

        results = {}
        key = self.robot_key[self.robot_list.index(robot)]
        entity_link = key.body_names
        link_pose = self._host_array(key, "body_link_pose_w")
        origins = self._host_env_origins() if is_relative else None

        if link_name not in entity_link:
            raise ValueError(f"Link name {link_name} not found in robot {robot.robot_name}")

        link_idx = entity_link.index(link_name)
        for env_idx in range(self.num_envs):
            if env_idx in env_idx_list:
                pose = np.array(link_pose[env_idx][link_idx], copy=True)
                if is_relative:
                    pose[:3] -= origins[env_idx]
                results[env_idx] = pose
            else:
                results[env_idx] = None
        return results

    def get_configured_collision_bbox_state(self, *, robot_index, env_idx):
        """Return a physical robot's local envelope and measured world XYZ/WXYZ.

        ``robot_index`` addresses ``config.robot.robots`` entries, so the two
        logical arms of a coupled robot share one configured envelope.
        """
        if isinstance(robot_index, bool) or not isinstance(robot_index, int | np.integer):
            raise TypeError(f"robot_index must be an integer, got {robot_index!r}.")
        robot_index = int(robot_index)
        if not 0 <= robot_index < len(self._configured_collision_bboxes):
            raise IndexError(
                f"robot_index {robot_index} is outside the configured physical robot range "
                f"[0, {len(self._configured_collision_bboxes)})."
            )
        if isinstance(env_idx, bool) or not isinstance(env_idx, int | np.integer):
            raise TypeError(f"env_idx must be an integer, got {env_idx!r}.")
        env_idx = int(env_idx)
        if not 0 <= env_idx < self.num_envs:
            raise IndexError(f"env_idx {env_idx} is outside the environment range [0, {self.num_envs}).")

        local_bbox = self._configured_collision_bboxes[robot_index]
        if local_bbox is None:
            raise RuntimeError(f"robot.robots[{robot_index}] has no configured collision_bbox.")
        physical_owners = [
            index
            for index, (_, owns_scene_robot) in enumerate(zip(self.robot_list, self.use_scene_cfg, strict=True))
            if owns_scene_robot
        ]
        if len(physical_owners) != len(self.robots_cfg):
            raise RuntimeError("Physical robot ownership is inconsistent with robot.robots.")
        articulation = self.robot_key[physical_owners[robot_index]]
        pose = np.array(self._host_array(articulation, "root_pose_w")[env_idx], dtype=float, copy=True).reshape(-1)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise RuntimeError(
                f"Robot {robot_index} env {env_idx} root pose must be finite XYZ+WXYZ; got shape {pose.shape}."
            )
        if np.linalg.norm(pose[3:]) <= np.finfo(float).eps:
            raise RuntimeError(f"Robot {robot_index} env {env_idx} root quaternion has zero length.")
        return local_bbox.copy(), pose[:3].copy(), pose[3:].copy()

    def get_joint(self, robot, env_idx_list=None):
        return self._get_joint_positions(robot, robot.arm_joint_indices, env_idx_list)

    def get_end_effector_real_val(self, robot, env_idx_list=None):
        return self._get_joint_positions(robot, robot.gripper_joint_indices, env_idx_list)

    def _get_joint_positions(self, robot, joint_indices, env_idx_list=None):
        articulation = self.robot_key[self.robot_list.index(robot)]
        env_indices = range(self.num_envs) if env_idx_list is None else env_idx_list
        results = dict.fromkeys(range(self.num_envs))
        joint_state = self._host_array(articulation, "joint_pos")
        for env_idx in env_indices:
            results[env_idx] = joint_state[env_idx, list(joint_indices)].copy()
        return results

    def get_waist_joint(self, robot, env_idx_list=None):
        return self._get_joint_positions(robot, getattr(robot, "waist_joint_indices", []), env_idx_list)

    def get_leg_joint(self, robot, env_idx_list=None):
        return self._get_joint_positions(robot, getattr(robot, "leg_joint_indices", []), env_idx_list)

    @staticmethod
    def planner_key(robot):
        return str(getattr(robot, "planner_name", robot.robot_name))

    def get_planner(self, robot=None):
        if robot is not None:
            return self.planner[self.planner_key(robot)]
        if len(self.planner) != 1:
            raise ValueError("robot is required unless the environment contains exactly one planner.")
        return next(iter(self.planner.values()))

    def get_named_joint_positions(self, joint_names, env_idx, robot=None):
        """Read measured joints in policy/planner name order."""
        if not self.robot_key:
            raise RuntimeError("Robot assets are not initialized.")
        robot = self.robot_list[0] if robot is None else robot
        articulation = self.robot_key[self.robot_list.index(robot)]
        names = list(articulation.joint_names)
        missing = [name for name in joint_names if name not in names]
        if missing:
            raise ValueError(f"Joints not found in articulation: {missing}.")
        indices = [names.index(name) for name in joint_names]
        return np.asarray(self._host_array(articulation, "joint_pos")[env_idx], dtype=np.float64)[indices].copy()

    def _canonical_action_joint_names(self, joint_names, robot):
        """Namespace only names shared by different physical articulations."""
        articulation = self.robot_key[self.robot_list.index(robot)]
        physical = []
        for candidate in self.robot_key:
            if not any(candidate is known for known in physical):
                physical.append(candidate)
        index = next(index for index, candidate in enumerate(physical) if candidate is articulation)
        return [
            f"robot{index}/{name}" if sum(name in candidate.joint_names for candidate in physical) > 1 else str(name)
            for name in joint_names
        ]

    def get_root_pose(self, robot=None, env_idx_list=None, is_relative=True):
        robot = self.robot_list[0] if robot is None else robot
        articulation = self.robot_key[self.robot_list.index(robot)]
        env_indices = range(self.num_envs) if env_idx_list is None else env_idx_list
        results = dict.fromkeys(range(self.num_envs))
        root_pose = self._host_array(articulation, "root_pose_w")
        origins = self._host_env_origins() if is_relative else None
        for env_idx in env_indices:
            pose = np.array(root_pose[env_idx], dtype=np.float64, copy=True)
            if is_relative:
                pose[:3] -= origins[env_idx]
            results[env_idx] = pose
        return results

    def teleport_root_pose(self, pose, env_idx=0):
        """Place the floating root in environment coordinates and clear motion."""
        value = np.asarray(pose, dtype=np.float64).reshape(-1)
        if value.shape != (7,) or not np.all(np.isfinite(value)):
            raise ValueError(f"root pose must be finite shape (7,), got {value.shape}.")
        quat_norm = np.linalg.norm(value[3:])
        if quat_norm <= 1e-12:
            raise ValueError("root pose quaternion must be non-zero.")
        value = value.copy()
        value[3:] /= quat_norm
        articulation = self.robot_key[0]
        value[:3] += np.asarray(self.scene.env_origins[env_idx].detach().cpu())
        root_state = articulation.data.root_state_w[env_idx : env_idx + 1].clone()
        root_state[0, :7] = torch.as_tensor(value, device=root_state.device, dtype=root_state.dtype)
        root_state[0, 7:13] = 0
        env_ids = torch.as_tensor([env_idx], device=articulation.device, dtype=torch.int32)
        self._host_state = None
        articulation.write_root_state_to_sim(root_state, env_ids=env_ids)
        adapter = self.mobile_base_adapter
        if adapter is not None:
            qw, qx, qy, qz = value[3:]
            yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            yaw_tensor = torch.as_tensor([yaw], device=adapter.device, dtype=torch.float32)
            adapter._yaw_controller.reset(env_ids, yaw_tensor)
            adapter._write_yaw_torque(env_ids, torch.zeros_like(yaw_tensor))
            adapter._target_velocities[env_idx] = 0.0
        self._last_applied_root_velocities.pop(int(env_idx), None)

    def get_base_link_pose(self, robot=None, env_idx_list=None, is_relative=True):
        robot = self.robot_list[0] if robot is None else robot
        return self.get_link_pose(robot, robot.base_link, env_idx_list, is_relative)

    def head_tool_frame(self, robot=None):
        robot = self.robot_list[0] if robot is None else robot
        frame = getattr(robot, "head_tool_frame", None)
        if frame is not None and frame not in self.get_planner(robot).tool_frames:
            raise KeyError(f"Head tool frame {frame!r} is not tracked by the planner.")
        return frame

    def get_real_head_pose(self, robot=None, env_idx_list=None, is_relative=True):
        """Read the measured head tool pose, including merged fixed-link offsets."""
        robot = self.robot_list[0] if robot is None else robot
        frame = self.head_tool_frame(robot)
        if frame is None:
            raise KeyError(f"Robot {robot.robot_name!r} has no head_tool_frame.")
        articulation = self.robot_key[self.robot_list.index(robot)]
        cache_key = (id(robot), id(articulation), frame)
        if self._head_pose_cache is None or self._head_pose_cache[0] != cache_key:
            from yourdfpy import URDF

            urdf = URDF.load(os.path.abspath(robot.urdf_path), build_scene_graph=True, load_meshes=False)
            if frame not in urdf.link_map:
                raise KeyError(f"Head tool frame {frame!r} is absent from {robot.urdf_path}.")
            body_frame, body_from_tool = nearest_body_frame(urdf, frame, articulation.body_names)
            self._head_pose_cache = (cache_key, body_frame, body_from_tool)
        _, body_frame, body_from_tool = self._head_pose_cache
        body_poses = self.get_link_pose(robot, body_frame, env_idx_list, is_relative)
        return {
            env_idx: None if pose is None else matrix_to_pose(pose_to_matrix(pose) @ body_from_tool)
            for env_idx, pose in body_poses.items()
        }

    def _poses_in_base_link(self, poses, robot=None, env_idx_list=None):
        base_poses = self.get_base_link_pose(robot, env_idx_list, is_relative=True)
        return {
            env_idx: (
                None
                if pose is None or base_poses[env_idx] is None
                else matrix_to_pose(np.linalg.inv(pose_to_matrix(base_poses[env_idx])) @ pose_to_matrix(pose))
            )
            for env_idx, pose in poses.items()
        }

    def get_base_endpose(self, robot, env_idx_list=None):
        return self._poses_in_base_link(self.get_real_endpose(robot, env_idx_list), robot, env_idx_list)

    def get_base_head_pose(self, robot=None, env_idx_list=None):
        return self._poses_in_base_link(self.get_real_head_pose(robot, env_idx_list), robot, env_idx_list)

    def get_root_velocity(self, robot=None, env_idx_list=None):
        """Read measured world-frame root-link linear and angular velocity."""
        robot = self.robot_list[0] if robot is None else robot
        articulation = self.robot_key[self.robot_list.index(robot)]
        results = dict.fromkeys(range(self.num_envs))
        for env_idx in range(self.num_envs) if env_idx_list is None else env_idx_list:
            results[env_idx] = as_numpy(articulation.data.root_link_state_w[env_idx, 7:13]).copy()
        return results

    def get_last_applied_root_velocity(self, env_idx):
        """Return the logical world-frame base command applied in this step."""
        velocity = self._last_applied_root_velocities.get(int(env_idx))
        return None if velocity is None else velocity.copy()

    def solve_tools(self, target_poses, env_idx, *, robot=None, control_group=None, trans="relative"):
        """Solve the configured tools together from measured whole-body joints."""
        planner = self.get_planner(robot)
        targets = planner.normalize_targets(target_poses)
        group = planner.resolve_control_group(
            list(targets),
            default_group=getattr(robot, "default_control_group", None) if robot is not None else None,
            group_name=control_group,
        )
        if trans == "relative":
            targets = {
                frame: self._trans_from_endlink_to_gripper(pose, planner.tool_robots[frame])
                if frame in planner.tool_robots
                else pose
                for frame, pose in targets.items()
            }
        elif trans != "world":
            raise ValueError(f"Unsupported transform mode: {trans}.")
        return planner.solve(
            current_qpos=self.get_named_joint_positions(planner.joints, env_idx, planner.owner),
            target_poses=targets,
            root_pose=self.get_root_pose(planner.owner, [env_idx], is_relative=True)[env_idx],
            control_group=group["name"],
            env_idx=env_idx,
        )

    def get_delta_endpose(self, robot, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = list(range(self.num_envs))
        delta_matrix = robot.delta_matrix
        real_endpose = self.get_real_endpose(robot, env_idx_list=env_idx_list, is_relative=True)
        results = {}
        for env_idx in range(self.num_envs):
            if env_idx in env_idx_list:
                ee_pose = real_endpose[env_idx]
                endpose_arr = np.eye(4)
                endpose_arr[:3, :3] = t3d.quaternions.quat2mat(ee_pose[-4:]) @ delta_matrix
                endpose_arr[:3, 3] = ee_pose[:3]
                res = endpose_arr[:3, 3].tolist() + t3d.quaternions.mat2quat(endpose_arr[:3, :3]).tolist()
                results[env_idx] = res
            else:
                results[env_idx] = None
        return results

    def get_robot_by_arm_name(self, arm_name):
        for robot in self.robot_list:
            if robot.arm_name == arm_name:
                return robot
        raise ValueError(f"No robot found with arm name: {arm_name}")

    def get_robot_by_gripper_name(self, gripper_name):
        for robot in self.robot_list:
            if robot.gripper_name == gripper_name:
                return robot
        raise ValueError(f"No robot found with gripper name: {gripper_name}")

    def plan_endeffector_joint(
        self,
        env_idx: int,
        arm_tag: str,
        target_val=None,
        result=None,
        need_plan: bool = True,
        control_step_num=100,
    ):
        if not need_plan:
            robot = self.get_robot_by_arm_name(arm_tag)
            control_info_list = []
            for i in range(result["position"].shape[0]):
                control_info = dict()
                control_info[self.process_name(robot.gripper_name)] = {
                    "position": result["position"][i],
                    "velocity": [0.0] * result["position"][i].shape[0],
                }
                control_info_list.append(control_info)
            return control_info_list

        robot = self.get_robot_by_arm_name(arm_tag)
        if robot.ee_type == "gripper":
            scale = robot.gripper_scale
            target_arr = np.asarray(target_val, dtype=float).reshape(-1)
            if target_arr.size == len(robot.gripper_joint_indices):
                real_val = target_arr.tolist()
            else:
                target_scalar = float(target_arr.reshape(-1)[0])
                val = (
                    target_scalar * (scale[1] - scale[0]) + scale[0]
                    if robot.gripper_move["sign"] == 1
                    else -target_scalar * (scale[1] - scale[0]) + scale[1]
                )
                real_val = [
                    val,
                    val * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2],
                ]
        elif robot.ee_type == "hand":
            real_val = np.asarray(target_val, dtype=float).reshape(-1).tolist()

        control_info_list = []
        now_endeffector_joint = self.get_end_effector_real_val(robot, env_idx_list=[env_idx])[env_idx]

        now_arr = np.asarray(now_endeffector_joint, dtype=float).reshape(-1)
        target_arr = np.asarray(real_val, dtype=float).reshape(-1)
        if now_arr.size == 1 and target_arr.size > 1:
            now_arr = np.full_like(target_arr, float(now_arr.item()))
        if now_arr.size != target_arr.size:
            raise ValueError(
                f"Endeffector joint size mismatch: now={now_arr.size}, target={target_arr.size} for {robot.robot_name}"
            )

        interp_n = int(control_step_num * 0.9)
        for step in range(control_step_num):
            if interp_n <= 1:
                pos = target_arr
            elif step < interp_n:
                alpha = step / float(interp_n - 1)
                pos = now_arr + (target_arr - now_arr) * alpha
            else:
                pos = target_arr
            control_info = dict()
            control_info[self.process_name(robot.gripper_name)] = {
                "position": pos.tolist(),
                "velocity": [0.0] * len(real_val),
            }
            control_info_list.append(control_info)

        return control_info_list

    def plan_ee(
        self,
        env_idx: int,
        arm_tag: str,
        result=None,
        need_plan: bool = False,
    ):
        if result is None or result["status"] != "Success":
            return None
        robot = self.get_robot_by_arm_name(arm_tag)
        control_info_list = []
        for i in range(result["position"].shape[0]):
            control_info = dict()
            control_info[self.process_name(robot.arm_name)] = {
                "position": result["position"][i],
                "velocity": result["velocity"][i],
            }
            control_info_list.append(control_info)
        return control_info_list

    def solve_ik(
        self,
        target_pose: List[float],
        env_idx: int,
        robot,
        trans: Literal["relative", "world"] = "world",
    ):
        now_qpos = self.get_joint(robot, env_idx_list=[env_idx])[env_idx]
        trans_target_pose = deepcopy(target_pose)
        if trans == "relative":
            trans_target_pose = self._trans_from_endlink_to_gripper(target_pose, robot)
        planner = self.ik_solver[robot.robot_name]
        robot_pose = deepcopy(robot.entity_origin_pose)
        return planner.solve_ik_to_joint(now_qpos, trans_target_pose, real_robot_pose=robot_pose)

    def process_name(self, name):
        if name.endswith("_state"):
            return name
        else:
            return name + "_joint_state"

    def restore_name(self, processed_name):
        if processed_name.endswith("_joint_state"):
            name = processed_name[:-12]
            if name.startswith("ee"):
                return "arm" + name[2:]
            else:
                return name
        else:
            return processed_name

    def control_robot(
        self,
        meta_control_list=None,
    ):
        self._last_applied_root_velocities.clear()
        self._apply_gravity_compensation()
        if meta_control_list is None:
            return
        sim = self.sim.sim
        scene = self.scene
        num_envs = scene.num_envs
        actions = {
            env_idx: control.get_action(self, env_idx)
            for env_idx, control in enumerate(meta_control_list)
            if isinstance(control, MetaControl)
        }
        named = {env_idx: action for env_idx, action in actions.items() if "qpos" in action or "qvel" in action}
        if named:
            self._control_direct_base(named)
            self._apply_named_joint_actions(named)
            for env_idx in named:
                self.control_manager.update_prev_control(env_idx, meta_control_list[env_idx])
        plan_num, plan_lst = 0, []
        for env_idx in range(num_envs):
            if env_idx in actions and env_idx not in named:
                plan_num += 1
                plan_lst.append(env_idx)
        if not plan_lst:
            return
        for robot in self.robot_list:
            if robot.robot_type == "arm":
                action_dim = len(robot.arm_joint_indices)
                gripper_dim = len(robot.gripper_joint_indices)
                arm = self.robot_key[self.robot_list.index(robot)]
                target_device = "cpu" if ensure_host_joint_targets(arm) is not None else sim.device
                arm_position = torch.zeros((plan_num, action_dim), device=target_device, dtype=torch.float32)
                arm_velocity = torch.zeros((plan_num, action_dim), device=target_device, dtype=torch.float32)
                gripper_position = torch.zeros((plan_num, gripper_dim), device=target_device, dtype=torch.float32)

                for id, env_idx in enumerate(plan_lst):
                    meta_control = meta_control_list[env_idx].get_action(
                        self,
                        env_idx=env_idx,
                    )  # get action dict
                    arm_position[id] = torch.tensor(
                        meta_control[self.process_name(robot.arm_name)]["position"],
                        device=target_device,
                        dtype=torch.float32,
                    )
                    arm_velocity[id] = torch.tensor(
                        meta_control[self.process_name(robot.arm_name)]["velocity"],
                        device=target_device,
                        dtype=torch.float32,
                    )
                    gripper_position[id][:] = torch.tensor(
                        meta_control[self.process_name(robot.gripper_name)]["position"],
                        device=target_device,
                        dtype=torch.float32,
                    )

                env_ids = torch.tensor(plan_lst, dtype=torch.int32, device=arm_velocity.device)
                arm.set_joint_position_target(arm_position, joint_ids=robot.arm_joint_indices, env_ids=env_ids)  # arm
                arm.set_joint_velocity_target(arm_velocity, joint_ids=robot.arm_joint_indices, env_ids=env_ids)  # arm
                arm.set_joint_position_target(
                    gripper_position,
                    joint_ids=robot.gripper_joint_indices,
                    env_ids=env_ids,
                )  # gripper
            else:
                pass

    def _apply_gravity_compensation(self):
        """Apply the source joint compensation once per physical articulation."""
        seen = set()
        for articulation in self.robot_key:
            if id(articulation) in seen:
                continue
            seen.add(id(articulation))
            if articulation.cfg.spawn.rigid_props.disable_gravity:
                continue
            forces = articulation.root_physx_view.get_gravity_compensation_forces()
            host_targets = ensure_host_joint_targets(articulation) is not None
            if host_targets:
                forces = as_numpy(forces, dtype=np.float32)
            else:
                forces = torch.as_tensor(forces, dtype=torch.float32, device=articulation.device)
            joint_count = len(articulation.joint_names)
            offset = forces.shape[1] - joint_count
            if offset < 0:
                raise RuntimeError("Gravity compensation force count is smaller than the articulation joint count.")
            effort = forces[:, offset : offset + joint_count]
            articulation.set_joint_effort_target(effort.copy() if host_targets else effort.clone())

    def _apply_named_joint_actions(self, actions):
        """Keep CUDA Eval targets on CPU until the articulation's batched upload."""
        articulations = []
        for articulation in self.robot_key:
            if any(articulation is known for known in articulations):
                continue
            articulation_index = len(articulations)
            articulations.append(articulation)
            host_targets = ensure_host_joint_targets(articulation) is not None
            for field, setter in (
                ("qpos", articulation.set_joint_position_target),
                ("qvel", articulation.set_joint_velocity_target),
            ):
                grouped = {}
                for env_idx, action in actions.items():
                    targets = action.get(field, {})
                    pairs = []
                    for joint_id, name in enumerate(articulation.joint_names):
                        if field == "qpos" and name in self.velocity_only_joint_names:
                            continue
                        key = f"robot{articulation_index}/{name}"
                        if key not in targets:
                            key = name
                        if key in targets:
                            pairs.append((joint_id, targets[key]))
                    ids = tuple(joint_id for joint_id, _ in pairs)
                    if ids:
                        grouped.setdefault(ids, []).append((env_idx, [value for _, value in pairs]))
                for joint_ids, rows in grouped.items():
                    values = [values for _, values in rows]
                    env_ids = [idx for idx, _ in rows]
                    if host_targets:
                        values = np.asarray(values, dtype=np.float32)
                        env_ids = np.asarray(env_ids, dtype=np.int64)
                    else:
                        values = torch.as_tensor(values, device=articulation.device, dtype=torch.float32)
                        env_ids = torch.as_tensor(env_ids, device=articulation.device, dtype=torch.int32)
                    setter(values, joint_ids=list(joint_ids), env_ids=env_ids)

    def _control_direct_base(self, actions):
        """Apply NavEval velocities, including its velocity-only yaw channel."""
        adapter = self.mobile_base_adapter
        yaw_name = None if adapter is None else adapter.yaw_velocity_joint_name
        commands = []
        for env_idx, action in actions.items():
            state = action.get(MetaControl.BASE_CONTROL_KEY)
            yaw_target = None if yaw_name is None else action.get("qvel", {}).get(yaw_name)
            if state is None and yaw_target is None:
                continue
            velocity = np.zeros(6) if state is None else np.asarray(state["velocity"], dtype=np.float64).copy()
            if yaw_target is not None:
                velocity[5] = float(yaw_target)
            commands.append((env_idx, velocity, None if state is None else state.get("target_yaw")))
        if not commands:
            return
        if adapter is None:
            raise RuntimeError("The configured robot has no mobile-base adapter.")
        target_yaws = None
        if any(yaw is not None for _, _, yaw in commands):
            target_yaws = np.asarray(
                [
                    float(t3d.euler.quat2euler(adapter.read_state(env_idx)[3:7])[2]) if yaw is None else float(yaw)
                    for env_idx, _, yaw in commands
                ]
            )
        adapter.apply_velocity(
            [env_idx for env_idx, _, _ in commands],
            np.stack([velocity for _, velocity, _ in commands]),
            target_yaws=target_yaws,
        )
        for env_idx, velocity, _ in commands:
            logical_velocity = np.zeros(6, dtype=np.float64)
            logical_velocity[:3] = velocity[:3]
            logical_velocity[5] = velocity[5]
            self._last_applied_root_velocities[int(env_idx)] = logical_velocity

    @property
    def velocity_only_joint_names(self):
        adapter = self.mobile_base_adapter
        return frozenset() if adapter is None else adapter.velocity_only_joint_names

    def _trans_from_endlink_to_gripper(self, target_pose, robot):
        inv_delta_matrix = robot.inv_delta_matrix
        target_pose_arr = np.array(target_pose)
        gripper_pose_pos, gripper_pose_quat = (
            deepcopy(target_pose_arr[0:3]),
            deepcopy(target_pose_arr[-4:]),
        )
        gripper_pose_mat = t3d.quaternions.quat2mat(gripper_pose_quat)
        gripper_pose_mat = gripper_pose_mat @ inv_delta_matrix
        gripper_pose_quat = t3d.quaternions.mat2quat(gripper_pose_mat)
        return list(gripper_pose_pos) + list(gripper_pose_quat)

    def reset(self):
        self._host_state = None
        self._last_applied_root_velocities.clear()
        self.control_manager.reset()
        if self.robot_key is None or len(self.robot_key) == 0:
            self._setup_robot_key()
        if self.mobile_base_adapter is not None:
            self.mobile_base_adapter.reset(range(self.num_envs))
        for robots in self._planner_setup_specs:
            self.get_planner(robots[0]).reset()
        self.robot_origin_endpose = [dict() for _ in range(self.num_envs)]
        self.robot_init_joint = [dict() for _ in range(self.num_envs)]
        self.set_robot_init_pose()

    def _load_camera_cfg(self, config: DictConfig):
        was_struct = OmegaConf.is_struct(config)
        OmegaConf.set_struct(config, False)

        def _get_robot_mount_name(i: int, idx: int):
            if not self.use_scene_cfg[i]:
                return f"robot{idx - 1}", idx
            return f"robot{idx}", idx + 1

        def _apply_camera_cfg(camera_cfg: dict, target_cfg: dict, robot_mount_name: str):
            target_cfg["mount_link"] = f"{robot_mount_name}/{camera_cfg['link']}"
            pos = camera_cfg.get("pos", None)
            if pos is not None:
                target_cfg["pos"] = pos
            ori = camera_cfg.get("ori", None)
            if ori is not None:
                target_cfg["ori"] = ori
            type = camera_cfg.get("type", None)
            if type is not None:
                target_cfg["type"] = type
            mesh = camera_cfg.get("mesh", None)
            if mesh is not None:
                target_cfg["mesh"] = mesh

        try:
            # wrist camera
            idx = 0
            for i, robot in enumerate(self.robot_list):
                if robot.type != "target":
                    continue
                robot_mount_name, idx = _get_robot_mount_name(i, idx)
                camera_cfg = getattr(robot, "camera", None)
                if camera_cfg is None:
                    continue

                for camera in camera_cfg:
                    name = camera.get("name", None)
                    output_name = camera.get("output_name", None)
                    if output_name is not None:
                        camera_name = str(output_name)
                    elif name.endswith("_wrist"):
                        if self.target_arm_nums == 1:
                            camera_name = name
                        elif self.target_arm_nums == 2:
                            camera_name = (
                                name.replace("cam_", "cam_left_", 1)
                                if i == 0
                                else name.replace("cam_", "cam_right_", 1)
                            )
                        else:
                            camera_name = f"{name}{i}"
                    elif not self.use_scene_cfg[i]:
                        continue
                    else:
                        camera_name = name

                    # Preserve cameras explicitly bound to existing USD prims.
                    if OmegaConf.select(config, f"{camera_name}.camera.prim_path") is not None:
                        continue

                    config[camera_name] = {
                        "camera": {},
                    }
                    _apply_camera_cfg(
                        camera,
                        config[camera_name]["camera"],
                        robot_mount_name,
                    )
        finally:
            OmegaConf.set_struct(config, was_struct)

    def _setup_robot_key(self):
        self._host_state = None
        idx = 0
        for i in range(len(self.robot_list)):
            if not self.use_scene_cfg[i]:
                self.robot_key.append(self.scene[f"robot{idx - 1}"])
            else:
                self.robot_key.append(self.scene[f"robot{idx}"])
                idx += 1
        for idx, robot in enumerate(self.robot_list):
            robot.gripper_joint_indices, gripper_names = self.robot_key[idx].find_joints(robot.gripper_joints_name)
            robot.gripper_joints_name = gripper_names

            if robot.robot_type == "arm":
                robot.arm_joint_indices, arm_names = self.robot_key[idx].find_joints(robot.arm_joints_name)
                robot.arm_joints_name = arm_names
                robot.action_dim = len(robot.arm_joint_indices)

            for names_attr, indices_attr in (
                ("waist_joints_name", "waist_joint_indices"),
                ("leg_joints_name", "leg_joint_indices"),
            ):
                names = list(getattr(robot, names_attr, []) or [])
                if names:
                    indices, resolved = self.robot_key[idx].find_joints(names, preserve_order=True)
                    setattr(robot, indices_attr, indices)
                    setattr(robot, names_attr, resolved)

            robot.base_link_origin_pose = self.get_link_pose(
                robot,
                link_name=robot.base_link,
                env_idx_list=[0],
                is_relative=True,
            )[0]
        self._setup_mobile_base_adapter()

    def _setup_mobile_base_adapter(self):
        adapter_cls = getattr(self.robot_list[0], "mobile_base_adapter_cls", None)
        if adapter_cls is not None:
            self.mobile_base_adapter = adapter_cls(
                articulation=self.robot_key[0], dt=self.dt, device=self.device, config=self.mobile_config
            )

    def initialize(self, sim: IsaacRLEnv):
        self._host_state = None
        self.sim = sim
        self.scene = sim.scene
        sim.refresh_host_state = self.refresh_host_state
        self.initialize_planners()

    def close(self):
        self._host_state = None
        self._last_applied_root_velocities.clear()
        self._head_pose_cache = None
        self.control_manager.reset()
        for robots in self._planner_setup_specs:
            planner = self.planner.pop(self.planner_key(robots[0]), None)
            if planner is not None:
                planner.close()
        self._planner_setup_requests = list(self._planner_setup_specs)
        self.robot_key.clear()
        self.mobile_base_adapter = None
        self.robot_origin_endpose = [dict() for _ in range(self.num_envs)]
        self.robot_init_joint = [dict() for _ in range(self.num_envs)]

    def _get_SceneCfg(self, num_envs, env_spacing, replicate_physics=False):
        scene_cfg = InteractiveSceneCfg(
            num_envs=num_envs,
            env_spacing=env_spacing,
            replicate_physics=replicate_physics,
        )
        for idx, robot in enumerate(self.robot_list):
            if self.use_scene_cfg[idx]:
                setattr(scene_cfg, f"robot{idx}", robot.SceneCfg)
        ensure_usd_path(scene_cfg)
        return scene_cfg

    def _setup_planner(self, robot):
        from env.planner_manager.curobo_planner import CuroboPlanner

        if robot.robot_type == "arm":
            root_pose = deepcopy(robot.entity_origin_pose)
            self.planner[robot.robot_name] = CuroboPlanner(
                robot_origin_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                active_joints_name=robot.arm_joints_name,
                all_joints=robot.arm_joints_name,
                dt=self.dt,
                yml_path=robot.curobo_yml_path,
                table_height=0.74 - root_pose[2],
            )
            self.ik_solver[robot.robot_name] = self.planner[robot.robot_name]

    def _setup_tool_ik(self, robots):
        from env.planner_manager.pink_planner import PinkPlanner
        from env.planner_manager.tool_ik import ToolIKPlanner

        owner = robots[0]
        cfg = owner.planner_config
        if cfg.get("ik_backend", cfg.get("backend")) != "pink":
            raise ValueError("Coupled evaluation currently requires the Pink IK backend.")
        joints = list(cfg["joints"])
        groups = owner.control_groups
        frames = list(
            dict.fromkeys(
                [
                    frame
                    for frame in cfg.get("tool_frames", [])
                    if any(frame in g["tool_frames"] for g in groups.values())
                ]
                + [frame for group in groups.values() for frame in group["tool_frames"]]
            )
        )
        pink_cfg = dict(cfg.get("pink", {}) or {})
        pink_cfg.update(self._pink_overrides)
        implementation = PinkPlanner(
            robot_origin_pose=[0, 0, 0, 1, 0, 0, 0],
            active_joints_name=joints,
            urdf_path=owner.urdf_path,
            mesh_dir=owner.mesh_dir,
            dt=self.dt,
            yml_path=os.path.join(owner.robot_file, cfg["config"]) if cfg.get("config") else owner.curobo_yml_path,
            tool_frames=frames,
            locked_joint_positions=owner.robot_args.get("non_arm_home", {}),
            solver=cfg.get("pink_solver"),
            position_cost=pink_cfg.get("position_cost", 2.0),
            orientation_cost=pink_cfg.get("orientation_cost", 2.0),
            frame_task_gain=pink_cfg.get("gain", 0.8),
            posture_task_cost=pink_cfg.get("posture_task_cost", 1e-2),
            null_space_task_cost=pink_cfg.get("null_space_task_cost", 1e-2),
            previous_command_task_cost=pink_cfg.get("previous_command_task_cost", 0.0),
            previous_command_reset_threshold=pink_cfg.get("previous_command_reset_threshold", 0.25),
            null_space_weight_overrides=pink_cfg.get("null_space_weight_overrides"),
            null_space_current_joints=pink_cfg.get("null_space_current_joints"),
            damping_task_cost=pink_cfg.get("damping_task_cost", 1e-4),
            max_joint_delta_per_step=cfg.get("pink_max_joint_delta_per_step", 0.8),
            self_collision_check=pink_cfg.get("self_collision_check", False),
        )
        self.planner[self.planner_key(owner)] = ToolIKPlanner(
            name=self.planner_key(owner),
            implementation=implementation,
            dt=self.dt,
            joints=joints,
            tool_frames=frames,
            owner=owner,
            tool_robots={robot.target_tool_frame: robot for robot in robots},
            control_groups=groups,
        )

    def _set_robot_obs_name(self):
        if self.target_arm_nums == 1:
            other_idx = 0
            for _, robot in enumerate(self.robot_list):
                if robot.type == "target":
                    robot.arm_name = "arm"
                    robot.gripper_name = "ee"
                else:
                    robot.arm_name = f"support_arm{other_idx}"
                    robot.gripper_name = f"support_ee{other_idx}"
                    other_idx += 1
        elif self.target_arm_nums == 2:
            target_idx, other_idx = 0, 0
            for _, robot in enumerate(self.robot_list):
                if robot.type == "target":
                    if target_idx == 0:
                        robot.arm_name = "left_arm"
                        robot.gripper_name = "left_ee"
                        target_idx += 1
                    elif target_idx == 1:
                        robot.arm_name = "right_arm"
                        robot.gripper_name = "right_ee"
                else:
                    robot.arm_name = f"support_arm{other_idx}"
                    robot.gripper_name = f"support_ee{other_idx}"
                    other_idx += 1
        else:
            target_idx, other_idx = 0, 0
            for _, robot in enumerate(self.robot_list):
                if robot.type == "target":
                    robot.arm_name = f"arm{target_idx}"
                    robot.gripper_name = f"ee{target_idx}"
                    target_idx += 1
                else:
                    robot.arm_name = f"support_arm{other_idx}"
                    robot.gripper_name = f"support_ee{other_idx}"
                    other_idx += 1

    def _import_symbol(self, module_path: str, symbol_name: str):
        module = import_module(module_path, package=__package__)
        return getattr(module, symbol_name)

    def _attach_scene_cfg(self, robot_name: str, robots, idx: int, cfg: dict):
        """
        Attach IsaacLab SceneCfg to robot instances.

        For single robot:
            robots[0] is the robot itself.

        For dual robots:
            robots[0] is left_robot.
            The scene init pose uses left_robot.entity_origin_pose by default.
        """
        if not isinstance(robots, tuple):
            robots = (robots,)

        scene_cfg = self._get_robot_cfg(robot_name=robot_name)

        # Default rule:
        # - single robot: use robot.entity_origin_pose
        # - dual robot: use left_robot.entity_origin_pose
        base_pose = robots[0].entity_origin_pose
        articulation_overrides = {}
        if cfg is not None and "enabled_self_collisions" in cfg:
            articulation_overrides["enabled_self_collisions"] = bool(cfg["enabled_self_collisions"])
        if cfg is not None and "fix_root_link" in cfg:
            articulation_overrides["fix_root_link"] = bool(cfg["fix_root_link"])
        if articulation_overrides:
            scene_cfg = scene_cfg.replace(
                spawn=scene_cfg.spawn.replace(
                    articulation_props=scene_cfg.spawn.articulation_props.replace(**articulation_overrides)
                )
            )

        scene_cfg = scene_cfg.replace(
            prim_path=f"{ENV_REGEX_NAMESPACE}/robot{idx}",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=base_pose[:3],
                rot=base_pose[-4:],
                joint_pos=scene_cfg.init_state.joint_pos,
            ),
        )

        for robot in robots:
            robot.SceneCfg = scene_cfg

        return robots[0] if len(robots) == 1 else robots

    def _get_robot(self, cfg: dict, idx: int):
        robot_name = cfg.get("robot_name", None)
        if robot_name is None:
            raise ValueError("robot_name must be specified")

        if robot_name not in ROBOT_CLASS_REGISTRY:
            raise ValueError(f"No such robot: {robot_name}")

        robot_info = ROBOT_CLASS_REGISTRY[robot_name]
        module_name = robot_info["module"]
        class_names = robot_info["classes"]

        module_path = f".robot_class.{module_name}"

        robot_classes = [self._import_symbol(module_path, class_name) for class_name in class_names]

        robots = tuple(robot_class(cfg=cfg) for robot_class in robot_classes)
        for robot in robots:
            if cfg.get("coupled", False):
                robot.planner_name = f"robot{idx}:{robot_name}"
            if cfg.get("planner"):
                robot.planner_config = {**getattr(robot, "planner_config", {}), **dict(cfg["planner"])}

        return self._attach_scene_cfg(
            robot_name=robot_name,
            robots=robots,
            idx=idx,
            cfg=cfg,
        )

    def _get_robot_cfg(self, robot_name=None):
        if robot_name is None:
            raise ValueError("robot_name must be specified")

        if robot_name not in ROBOT_CONFIG_REGISTRY:
            raise ValueError(f"No such robot: {robot_name}")

        module_name = ROBOT_CONFIG_REGISTRY[robot_name]
        module = import_module(f".robot_config.{module_name}", package=__package__)

        return module.get_robot_config()


ROBOT_CLASS_REGISTRY = {
    "tianji_marvin_wuji_left": {
        "module": "tianji_marvin_wuji",
        "classes": ("LeftTianjiMarvinWuji",),
    },
    "tianji_marvin_wuji_right": {
        "module": "tianji_marvin_wuji",
        "classes": ("RightTianjiMarvinWuji",),
    },
    "m92uw": {
        "module": "m92uw",
        "classes": ("LeftM92UW", "RightM92UW"),
    },
    "franka": {
        "module": "franka",
        "classes": ("Franka",),
    },
    "x5": {
        "module": "x5",
        "classes": ("X5",),
    },
}


ROBOT_CONFIG_REGISTRY = {
    "tianji_marvin_wuji_left": "tianji_marvin_wuji_left",
    "tianji_marvin_wuji_right": "tianji_marvin_wuji_right",
    "m92uw": "m92uw",
    "franka": "franka",
    "x5": "x5",
}
