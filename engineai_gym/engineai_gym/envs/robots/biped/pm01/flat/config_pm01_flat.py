from engineai_gym.envs.robots.biped.pm01.rough.config_pm01_rough import (
    ConfigPm01Rough,
)


class ConfigPm01Flat(ConfigPm01Rough):
    class env(ConfigPm01Rough.env):
        obs_list = [
            "base_lin_vel",
            "pos_phase",
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
        ]

    class terrain(ConfigPm01Rough.terrain):
        mesh_type = "plane"
        measure_heights = False
