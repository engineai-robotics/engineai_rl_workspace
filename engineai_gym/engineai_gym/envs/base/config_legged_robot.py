from engineai_rl_lib.base_config import BaseConfig


class ConfigLeggedRobot(BaseConfig):
    class env:
        num_envs = 4096
        # obs to save in obs_dict
        obs_list = [
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "dof_pos",
            "dof_vel",
            "actions",
            "height_measurements",
        ]
        # goals to save in goal_dict
        goal_list = ["commands"]
        # joints used for action
        action_joints = ["joint_a", "joint_b"]
        # not used with heightfields/trimeshes
        env_spacing = 3.0
        # send time out information to the algorithm
        send_timeouts = True
        # episode length in seconds
        episode_length_s = 20
        # tester config used in play
        tester_config_path = "{ENGINEAI_GYM_PACKAGE_DIR}/tester/tester_config.yaml"

    class safety:
        # a multiplier applied on torque limit of each joint, as a dict: {"joint_name": multiplier}. None if maintaining original torque limit.
        torque_hard_limit_multi = None

    class terrain:
        # None, plane, heightfield or trimesh
        mesh_type = "trimesh"
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        # rough terrain only:
        measure_heights = True
        measured_points_x = [
            -0.8,
            -0.7,
            -0.6,
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
        ]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        # select a unique terrain type and pass all arguments
        selected = False
        # Dict of arguments for selected terrain
        terrain_kwargs = None
        # starting curriculum state
        max_init_terrain_level = 5
        terrain_length = 8.0
        terrain_width = 8.0
        # number of terrain rows (levels)
        num_rows = 10
        # number of terrain cols (types)
        num_cols = 20
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        # slopes above this threshold will be corrected to vertical surfaces
        slope_threshold = 0.75

    class commands:
        curriculum = False
        max_curriculum = 1.0
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw
        num_commands = 3
        # time before command are changed[s]
        resampling_time = 10.0
        # if true: compute ang vel command from heading error
        yaw_from_heading_target = True
        # probability of still commands
        still_ratio = 0.25
        # set command to zero if command < set_zero_threshold
        set_zero_threshold = 0.3

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 1.0]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        # target joint angles when action = 0.0
        default_joint_angles = {"joint_a": 0.0, "joint_b": 0.0}

    class control:
        # P: position, V: velocity, T: torques
        control_type = "P"
        # PD Drive parameters:
        stiffness = {"joint_a": 10.0, "joint_b": 15.0}  # [N*m/rad]
        damping = {"joint_a": 1.0, "joint_b": 1.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scales = {"joint_a": 0.5, "joint_b": 0.5}
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        # path of URDF file
        file = ""
        # actor name
        name = "legged_robot"
        # name of the foot bodies, used to index body state and contact force tensors
        foot_name = None
        # half size of foot on z-axis
        foot_half_z_size = 0.05
        # name of the knee bodies, used to index body state and contact force tensors
        knee_name = None
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        collapse_fixed_joints = True
        # fixe the base of the robot
        fix_base_link = False
        # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        default_dof_drive_mode = 3
        # 1 to disable, 0 to enable...bitwise filter
        self_collisions = 0
        # replace collision cylinders with capsules, leads to faster/more stable simulation
        replace_cylinder_with_capsule = True
        # Some .obj meshes must be flipped from y-up to z-up
        flip_visual_attachments = True
        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01
        # minimal joint armature for all joints, since too small armature may cause robots float
        min_joint_armature = 0.001
        joint_armature = {"joint_a": 0.03, "joint_b": 0.0025}
        joint_friction = {"joint_a": 1, "joint_b": 1}

    class gait:
        # full cycle time
        cycle_time = 0.8

    class domain_rands:
        class rigid_shape:
            randomize_friction = True
            friction_range = [0.5, 1.25]
            randomize_restitution = True
            restitution_range = [0.0, 0.4]

        class rigid_body:
            randomize_base_mass = False
            added_mass_range = [-2.5, 2.5]
            randomize_com = False
            com_displacement_range = [-0.05, 0.05]
            randomize_link_mass = False
            link_mass_multi_range = [0.9, 1.1]

        class dof:
            randomize_gains = False
            stiffness_multi_range = [0.8, 1.2]
            damping_multi_range = [0.8, 1.2]
            randomize_torque = False
            torque_multip_range = [0.8, 1.2]
            randomize_motor_offset = False
            motor_offset_range = [-0.035, 0.035]
            randomize_joint_friction = False
            randomize_joint_friction_each_joint = False
            joint_friction_multi_range = [0.01, 1.15]
            joint_friction_multi_range_each_joint = {
                "joint_a": [0.01, 1.15],
                "joint_b": [0.5, 1.3],
            }
            randomize_joint_armature = False
            randomize_joint_armature_each_joint = False
            joint_armature_multi_range = [-0.03, 0.03]
            joint_armature_multi_range_each_joint = {
                "joint_a": [-0.03, 0.03],
                "joint_b": [-0.03, 0.03],
            }
            randomize_coulomb_friction = False
            joint_coulomb_range = [0.1, 0.9]
            joint_viscous_range = [0.05, 0.1]

        class action_lag:
            # if action_lag_timesteps > 1 and randomize_action_lag_timesteps = False, the action_lag_timesteps is fixed.
            # if randomize_action_lag_timesteps = True and randomize_action_lag_timesteps_perstep = False, action_lag_timesteps will be randomized on reset
            # if randomize_action_lag_timesteps = True and randomize_action_lag_timesteps_perstep = True, action_lag_timesteps will be randomized on each step
            action_lag_timesteps = 0
            randomize_action_lag_timesteps = False
            randomize_action_lag_timesteps_perstep = False
            action_lag_timesteps_range = [2, 4]

        class obs_lag:
            # if motor_lag_timesteps > 1 and randomize_motor_lag_timesteps = False, the motor_lag_timesteps is fixed.
            # if randomize_motor_lag_timesteps = True and randomize_motor_lag_timesteps_perstep = False, motor_lag_timesteps will be randomized on reset
            # if randomize_motor_lag_timesteps = True and randomize_motor_lag_timesteps_perstep = True, motor_lag_timesteps will be randomized on each step
            motor_lag_timesteps = 0
            randomize_motor_lag_timesteps = False
            randomize_motor_lag_timesteps_perstep = False
            motor_lag_timesteps_range = [2, 10]
            # if imu_lag_timesteps > 1 and randomize_imu_lag_timesteps = False, the imu_lag_timesteps is fixed.
            # if randomize_imu_lag_timesteps = True and randomize_imu_lag_timesteps_perstep = False, imu_lag_timesteps will be randomized on reset
            # if randomize_imu_lag_timesteps = True and randomize_imu_lag_timesteps_perstep = True, imu_lag_timesteps will be randomized on each step
            imu_lag_timesteps = 0
            randomize_imu_lag_timesteps = False
            randomize_imu_lag_timesteps_perstep = False
            imu_lag_timesteps_range = [1, 2]

        class disturbance:
            push_robots = True
            push_interval_s = 15
            max_push_vel_xy = 1.0
            max_push_ang_vel = 0.6

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.0
            torques = -1e-05
            dof_vel = -0.0
            dof_acc = -2.5e-07
            base_height = -0.0
            feet_air_time = 1.0
            collision = -1.0
            feet_stumble = -0.0
            action_rate = -0.01
            stand_still = -0.0
            feet_distance = 0.2
            foot_slip = -0.1
            base_acc = 0.2
            vel_mismatch_exp = 0.5
            track_vel_hard = 0.5
            default_joint_pos = 0.8
            low_speed = 0.2
            action_smoothness = -0.003

        class params:
            # if true negative total rewards are clipped at zero (avoids early termination problems)
            only_positive_rewards = True
            # tracking reward = exp(-error^2/sigma)
            tracking_sigma = 0.25
            # multiplier of URDF limits, values above this limit are penalized
            soft_dof_pos_limit_multi = 1.0
            soft_dof_vel_limit_multi = 1.0
            soft_torque_limit_multi = {
                "joint": 1.0,
            }
            base_height_target = 1.0
            # forces above this value are penalized
            max_contact_force = 100.0
            min_feet_dist = 0.15
            max_feet_dist = 0.8
            target_feet_height = 0.2

    class normalization:
        obs_scales = {
            "base_lin_vel": 2.0,
            "base_ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "height_measurements": 5.0,
            "body_mass": 0.1,
        }
        clip_actions = 100.0

    class contact:
        foot_contact_threshold = 5.0

    class viewer:
        ref_env = 0
        pos = [10, 0, 6]
        lookat = [11.0, 5, 3.0]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2
