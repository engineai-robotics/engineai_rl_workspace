from engineai_gym.envs.robots.quadruped.a1.rough.config_a1_rough import ConfigA1Rough


class ConfigA1Flat(ConfigA1Rough):
    class env(ConfigA1Rough.env):
        obs_list = [
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "dof_pos",
            "dof_vel",
            "actions",
        ]

    class terrain(ConfigA1Rough.terrain):
        mesh_type = "plane"
        measure_heights = False
