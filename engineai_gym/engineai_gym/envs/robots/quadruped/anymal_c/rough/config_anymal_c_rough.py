from engineai_gym.envs.base.config_legged_robot import ConfigLeggedRobot


class ConfigAnymalCRough(ConfigLeggedRobot):
    class env(ConfigLeggedRobot.env):
        num_envs = 64
        action_joints = [
            "LF_HAA",
            "RF_HAA",
            "RH_HAA",
            "RH_HAA",
            "LF_HFE",
            "LH_HFE",
            "RF_HFE",
            "RH_HFE",
            "LF_KFE",
            "LH_KFE",
            "RF_KFE",
            "RH_KFE",
        ]
        tester_config_path = "{ENGINEAI_GYM_PACKAGE_DIR}/envs/robots/quadruped/anymal_c/rough/tester_config.yaml"

    class terrain(ConfigLeggedRobot.terrain):
        mesh_type = "trimesh"

    class init_state(ConfigLeggedRobot.init_state):
        pos = [0.0, 0.0, 0.6]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "LF_HAA": 0.0,
            "LH_HAA": 0.0,
            "RF_HAA": -0.0,
            "RH_HAA": -0.0,
            "LF_HFE": 0.4,
            "LH_HFE": -0.4,
            "RF_HFE": 0.4,
            "RH_HFE": -0.4,
            "LF_KFE": -0.8,
            "LH_KFE": 0.8,
            "RF_KFE": -0.8,
            "RH_KFE": 0.8,
        }

    class control(ConfigLeggedRobot.control):
        # PD Drive parameters:
        stiffness = {"HAA": 80.0, "HFE": 80.0, "KFE": 80.0}  # [N*m/rad]
        damping = {"HAA": 2.0, "HFE": 2.0, "KFE": 2.0}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scales = {"HAA": 0.5, "HFE": 0.5, "KFE": 0.5}
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = False
        actuator_net_file = (
            "{ENGINEAI_GYM_PACKAGE_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"
        )

    class asset(ConfigLeggedRobot.asset):
        file = "{ENGINEAI_GYM_PACKAGE_DIR}/resources/robots/quadruped/anymal_c/urdf/anymal_c.urdf"
        name = "anymal_c"
        foot_name = "FOOT"
        penalize_contacts_on = ["SHANK", "THIGH"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        joint_armature = {"HAA": 0.04, "HFE": 0.04, "KFE": 0.04}
        joint_friction = {"HAA": 1, "HFE": 1, "KFE": 1}

    class domain_rands(ConfigLeggedRobot.domain_rands):
        class rigid_body(ConfigLeggedRobot.domain_rands.rigid_body):
            randomize_base_mass = True
            added_mass_range = [-5.0, 5.0]

    class rewards(ConfigLeggedRobot.rewards):
        class params(ConfigLeggedRobot.rewards.params):
            base_height_target = 0.5
            max_contact_force = 500.0
            only_positive_rewards = True
            soft_torque_limit_multi = {"HAA": 1.0, "HFE": 1.0, "KFE": 1.0}

        class scales(ConfigLeggedRobot.rewards.scales):
            pass
