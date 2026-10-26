from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

from env.global_configs import ROBOTS_PATH

_HAND_JOINTS = [
    "r_thumb_cmc_flex",
    "r_thumb_cmc_abd",
    "r_thumb_mcp",
    "r_thumb_ip",
    "r_index_finger_mcp_flex",
    "r_index_finger_mcp_abd",
    "r_index_finger_pip",
    "r_index_finger_dip",
    "r_middle_finger_mcp_flex",
    "r_middle_finger_mcp_abd",
    "r_middle_finger_pip",
    "r_middle_finger_dip",
    "r_ring_finger_mcp_flex",
    "r_ring_finger_mcp_abd",
    "r_ring_finger_pip",
    "r_ring_finger_dip",
    "r_pinky_mcp_flex",
    "r_pinky_mcp_abd",
    "r_pinky_pip",
    "r_pinky_dip",
]


def get_robot_config() -> ArticulationCfg:
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ROBOTS_PATH}/tianji_marvin_wuji/tianji_marvin_wuji_right.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                stabilization_threshold=0.001,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "Joint1_R": -2.09,
                "Joint2_R": -1.57,
                "Joint3_R": 1.57,
                "Joint4_R": -2.09,
                "Joint5_R": 0.0,
                "Joint6_R": 0.0,
                "Joint7_R": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["Joint[1-7]_R"],
                effort_limit_sim=300.0,
                velocity_limit_sim=300.0,
                stiffness=8000.0,
                damping=50.0,
                armature=0.05,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=_HAND_JOINTS,
                effort_limit_sim=50.0,
                velocity_limit_sim=50.0,
                stiffness=800.0,
                damping=20.0,
                armature=0.01,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
