from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

from env.global_configs import ROBOTS_PATH

_HAND_JOINTS = [
    "l_thumb_cmc_flex",
    "l_thumb_cmc_abd",
    "l_thumb_mcp",
    "l_thumb_ip",
    "l_index_finger_mcp_flex",
    "l_index_finger_mcp_abd",
    "l_index_finger_pip",
    "l_index_finger_dip",
    "l_middle_finger_mcp_flex",
    "l_middle_finger_mcp_abd",
    "l_middle_finger_pip",
    "l_middle_finger_dip",
    "l_ring_finger_mcp_flex",
    "l_ring_finger_mcp_abd",
    "l_ring_finger_pip",
    "l_ring_finger_dip",
    "l_pinky_mcp_flex",
    "l_pinky_mcp_abd",
    "l_pinky_pip",
    "l_pinky_dip",
]


def get_robot_config() -> ArticulationCfg:
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ROBOTS_PATH}/tianji_marvin_wuji/tianji_marvin_wuji_left.usd",
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
                "Joint1_L": 2.09,
                "Joint2_L": -1.57,
                "Joint3_L": -1.57,
                "Joint4_L": -2.09,
                "Joint5_L": 0.0,
                "Joint6_L": 0.0,
                "Joint7_L": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["Joint[1-7]_L"],
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
