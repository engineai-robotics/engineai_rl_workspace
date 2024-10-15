from engineai_gym.envs.robots.biped.config_biped_robot import ConfigBipedRobot


class ConfigPm01Rough(ConfigBipedRobot):
    class env(ConfigBipedRobot.env):
        env_spacing = 1.5
        action_joints = [
            "j00_hip_pitch_l",
            "j01_hip_roll_l",
            "j02_hip_yaw_l",
            "j03_knee_pitch_l",
            "j04_ankle_pitch_l",
            "j05_ankle_roll_l",
            "j06_hip_pitch_r",
            "j07_hip_roll_r",
            "j08_hip_yaw_r",
            "j09_knee_pitch_r",
            "j10_ankle_pitch_r",
            "j11_ankle_roll_r",
        ]

        obs_list = [
            "base_lin_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "dof_pos_ref_diff",
            "base_ang_vel",
            "base_euler_xyz",
            "rand_push_force",
            "rand_push_torque",
            "terrain_frictions",
            "body_mass",
            "stance_curve",
            "swing_curve",
            "contact_mask",
            "height_measurements",
        ]
        goal_list = ["pos_phase", "commands"]
        use_ref_actions = False
        episode_length_s = 24

    class terrain(ConfigBipedRobot.terrain):
        static_friction = 0.6
        dynamic_friction = 0.6
        num_rows = 20  # number of terrain rows (levels)
        max_init_terrain_level = 10
        terrain_proportions = [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]

    class init_state(ConfigBipedRobot.init_state):
        pos = [0.0, 0.0, 0.9]

        default_joint_angles = {
            "j00_hip_pitch_l": -0.24,
            "j01_hip_roll_l": 0.0,
            "j02_hip_yaw_l": 0.0,
            "j03_knee_pitch_l": 0.48,
            "j04_ankle_pitch_l": -0.24,
            "j05_ankle_roll_l": 0.0,
            "j06_hip_pitch_r": -0.24,
            "j07_hip_roll_r": 0.0,
            "j08_hip_yaw_r": 0.0,
            "j09_knee_pitch_r": 0.48,
            "j10_ankle_pitch_r": -0.24,
            "j11_ankle_roll_r": 0.0,
        }

    class control(ConfigBipedRobot.control):
        # PD Drive parameters:
        control_type = "P"

        stiffness = {
            "hip_pitch": 70,
            "hip_roll": 50,
            "hip_yaw": 50,
            "knee_pitch": 70,
            "ankle_pitch": 20,
            "ankle_roll": 20,
        }

        damping = {
            "hip_pitch": 7.0,
            "hip_roll": 5.0,
            "hip_yaw": 5.0,
            "knee_pitch": 7.0,
            "ankle_pitch": 0.2,
            "ankle_roll": 0.2,
        }

        # action scale: target angle = actionScale * action + defaultAngle
        action_scales = {
            "hip_pitch": 0.5,
            "hip_roll": 0.5,
            "hip_yaw": 0.5,
            "knee_pitch": 0.5,
            "ankle_pitch": 0.5,
            "ankle_roll": 0.5,
        }
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz

    class domain_rands(ConfigBipedRobot.domain_rands):
        class rigid_shape(ConfigBipedRobot.domain_rands.rigid_shape):
            randomize_friction = True
            friction_range = [0.2, 1.3]
            randomize_restitution = True
            restitution_range = [0.0, 0.4]

        class rigid_body(ConfigBipedRobot.domain_rands.rigid_body):
            randomize_base_mass = True
            added_mass_range = [-4.0, 4.0]

            randomize_com = True
            com_displacement_range = [-0.06, 0.06]

            randomize_link_mass = True
            link_mass_multi_range = [0.8, 1.2]

        class dof(ConfigBipedRobot.domain_rands.dof):
            randomize_gains = True
            stiffness_multi_range = [0.8, 1.2]
            damping_multi_range = [0.8, 1.2]

            randomize_torque = True
            torque_multip_range = [0.8, 1.2]

            randomize_motor_offset = True
            motor_offset_range = [-0.035, 0.035]

            randomize_joint_friction = True
            randomize_joint_friction_each_joint = True
            joint_friction_multi_range = [0.01, 1.15]
            joint_friction_multi_range_each_joint = {
                "hip": [0.01, 1.15],
                "knee": [0.01, 1.15],
                "ankle": [0.5, 1.3],
            }

            randomize_joint_armature = True
            randomize_joint_armature_each_joint = True
            joint_armature_multi_range = [0.27, 2]
            joint_armature_multi_range_each_joint = {
                "hip": [0.27, 2],
                "knee": [0.27, 2],
                "ankle": [0.27, 2],
            }

            randomize_coulomb_friction = False
            joint_coulomb_range = [0.1, 0.9]
            joint_viscous_range = [0.05, 0.1]

        class action_lag(ConfigBipedRobot.domain_rands.action_lag):
            action_lag_timesteps = 0
            randomize_action_lag_timesteps = True
            randomize_action_lag_timesteps_perstep = False
            action_lag_timesteps_range = [2, 5]

        class obs_lag(ConfigBipedRobot.domain_rands.obs_lag):
            motor_lag_timesteps = 0
            randomize_motor_lag_timesteps = True
            randomize_motor_lag_timesteps_perstep = False
            motor_lag_timesteps_range = [5, 15]
            imu_lag_timesteps = 0
            randomize_imu_lag_timesteps = True
            randomize_imu_lag_timesteps_perstep = False
            imu_lag_timesteps_range = [1, 10]

        class disturbance(ConfigBipedRobot.domain_rands.disturbance):
            push_robots = True
            push_interval_s = 8
            max_push_vel_xy = 0.4
            max_push_ang_vel = 0.6

    class asset(ConfigBipedRobot.asset):
        file = "{ENGINEAI_GYM_PACKAGE_DIR}/resources/robots/biped/pm01/urdf/pm01_only_legs_simple_collision.urdf"
        name = "pm01"
        foot_name = "link_ankle_roll"
        knee_name = "link_knee_pitch"

        terminate_after_contacts_on = [
            "link_base",
            "link_knee_pitch",
            "shoulder",
            "elbow",
            "torso",
        ]
        penalize_contacts_on = ["link_base"]
        foot_half_z_size = 0.0463
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        joint_armature = {"hip": 0.03, "knee": 0.03, "ankle": 0.0025}
        joint_friction = {"hip": 0, "knee": 0, "ankle": 0}

    class commands(ConfigBipedRobot.commands):
        curriculum = True
        max_curriculum = 1.7
        resampling_time = 8.0  # time before command are changed[s]
        yaw_from_heading_target = (
            False  # if true: compute ang vel command from heading error
        )
        num_commands = 3
        still_ratio = 0

        class ranges(ConfigBipedRobot.commands.ranges):
            lin_vel_x = [-1.5, 1.5]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]  # min max [m/s]
            ang_vel_yaw = [-1.5, 1.5]  # min max [rad/s]

    class normalization(ConfigBipedRobot.normalization):
        obs_scales = {
            "base_lin_vel": 2.0,
            "base_ang_vel": 1.0,
            "body_mass": 0.1,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "base_euler_xyz": 1.0,
            "height_measurements": 5.0,
        }

    class rewards(ConfigBipedRobot.rewards):
        class params(ConfigBipedRobot.rewards.params):
            base_height_target = 0.8132
            max_contact_force = 500.0
            tracking_sigma = 5
            target_joint_pos_scale = 0.26
            min_feet_dist = 0.15
            max_feet_dist = 0.8
            target_feet_height = 0.1
            soft_torque_limit_multi = {
                "hip": 0.85,
                "knee": 0.85,
                "ankle": 0.85,
                "shoulder": 0.85,
                "elbow": 0.85,
            }

        class scales(ConfigBipedRobot.rewards.scales):
            torques = -1e-10
            lin_vel_z = -0.0
            feet_air_time = 1.5
            dof_pos_limits = -0.0
            feet_contact_forces = -0.02
            tracking_lin_vel = 1.4
            tracking_ang_vel = 1.1
            dof_vel = -1e-5
            dof_acc = -5e-9
            orientation = 1.0
            base_height = 0.2

            termination = -0.0
            no_fly = 0.0
            ang_vel_xy = -0.0
            action_rate = -0.0
            stand_still = -0.0

            dof_ref_pos_diff = 2.2
            feet_distance = 0.2
            knee_distance = 0.2
            foot_slip = -0.1
            base_acc = 0.2
            vel_mismatch_exp = 0.5
            track_vel_hard = 0.5
            default_joint_pos = 0.8
            feet_height = -0.0
            low_speed = 0.2
            action_smoothness = -0.003
            feet_contact_number = 1.4
            feet_clearance = 1.6

    class sim(ConfigBipedRobot.sim):
        dt = 0.001  # 1000 Hz
