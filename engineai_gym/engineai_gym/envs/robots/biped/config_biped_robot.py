from engineai_gym.envs.base.config_legged_robot import ConfigLeggedRobot


class ConfigBipedRobot(ConfigLeggedRobot):
    class rewards(ConfigLeggedRobot.rewards):
        class params(ConfigLeggedRobot.rewards.params):
            target_joint_pos_scale = 0.26

        class scales(ConfigLeggedRobot.rewards.scales):
            dof_ref_pos_diff = 2.2
            knee_distance = 0.3
            feet_contact_number = 1.0
            feet_clearance = 1.0
