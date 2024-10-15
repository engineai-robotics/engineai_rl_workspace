from .config_a1_flat import ConfigA1Flat
from engineai_gym.envs.base.config_legged_robot_ref import ConfigLeggedRobotRef


class ConfigA1FlatRefState(ConfigA1Flat):
    class env(ConfigA1Flat.env):
        obs_list = [
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "dof_pos",
            "absolute_dof_pos",
            "dof_vel",
            "actions",
            "foot_pos",
            "base_z_pos",
        ]

    class ref_state(ConfigLeggedRobotRef.ref_state):
        ref_state_loader = True
        ref_state_init = True
        motion_files_path = (
            "{ENGINEAI_GYM_PACKAGE_DIR}/resources/robots/quadruped/a1/mocap_motions"
        )
        ref_state_init_prob = 0.85
        data_mapping = {
            "base_pos": "root_pos",
            "base_z_pos": "root_z_pos",
            "base_rot": "root_rot",
            "absolute_dof_pos": "dof_pos",
            "dof_vel": "dof_vel",
            "foot_pos": "foot_pos_local",
            "base_lin_vel": "root_lin_vel",
            "base_ang_vel": "root_ang_vel",
        }

    class init_state(ConfigA1Flat.init_state):
        default_joint_angles = {
            "FL_hip_joint": -0.15,
            "RL_hip_joint": -0.15,
            "FR_hip_joint": 0.15,
            "RR_hip_joint": 0.15,
            "FL_thigh_joint": 0.55,
            "RL_thigh_joint": 0.7,
            "FR_thigh_joint": 0.55,
            "RR_thigh_joint": 0.7,
            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }

    class control(ConfigA1Flat.control):
        stiffness = {"joint": 80.0}
        damping = {"joint": 1.0}
        decimation = 6

    class asset(ConfigA1Flat.asset):
        terminate_after_contacts_on = [
            "base",
            "FL_calf",
            "FR_calf",
            "RL_calf",
            "RR_calf",
            "FL_thigh",
            "FR_thigh",
            "RL_thigh",
            "RR_thigh",
        ]
        self_collisions = 0

    class domain_rands(ConfigA1Flat.domain_rands):
        class rigid_shape(ConfigA1Flat.domain_rands.rigid_shape):
            friction_range = [0.25, 1.75]

        class rigid_body(ConfigA1Flat.domain_rands.rigid_body):
            randomize_base_mass = True
            added_mass_range = [-1.0, 1.0]

        class dof(ConfigA1Flat.domain_rands.dof):
            randomize_gains = True
            stiffness_multi_range = [0.9, 1.1]
            damping_multi_range = [0.9, 1.1]

    class commands(ConfigA1Flat.commands):
        yaw_from_heading_target = False
        still_ratio = 0

        class ranges(ConfigA1Flat.commands.ranges):
            lin_vel_x = [-1.0, 2.0]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-1.57, 1.57]

    class rewards(ConfigA1Flat.rewards):
        class params(ConfigA1Flat.rewards.params):
            soft_dof_pos_limit_multi = 0.9
            base_height_target = 0.25

        class scales(ConfigA1Flat.rewards.scales):
            termination = -0.0
            tracking_lin_vel = 1.5 * 1.0 / (0.005 * 6)
            tracking_ang_vel = 0.5 * 1.0 / (0.005 * 6)
            lin_vel_z = -0.0
            ang_vel_xy = -0.0
            orientation = -0.0
            torques = -0.0
            dof_vel = -0.0
            dof_acc = -0.0
            base_height = -0.0
            feet_air_time = 0.0
            collision = -0.0
            feet_stumble = -0.0
            action_rate = -0.0
            stand_still = -0.0
            feet_distance = 0.0
            foot_slip = -0.0
            base_acc = 0.0
            vel_mismatch_exp = 0.0
            track_vel_hard = 0.0
            default_joint_pos = 0.0
            low_speed = 0.0
            action_smoothness = -0.0
