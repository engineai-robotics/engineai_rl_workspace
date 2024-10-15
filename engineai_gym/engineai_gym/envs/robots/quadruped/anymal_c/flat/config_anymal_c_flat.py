from engineai_gym.envs.robots.quadruped.anymal_c.rough.config_anymal_c_rough import (
    ConfigAnymalCRough,
)


class ConfigAnymalCFlat(ConfigAnymalCRough):
    class terrain(ConfigAnymalCRough.terrain):
        mesh_type = "plane"
        measure_heights = False

    class asset(ConfigAnymalCRough.asset):
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(ConfigAnymalCRough.rewards):
        class params(ConfigAnymalCRough.rewards.params):
            max_contact_force = 350.0

        class scales(ConfigAnymalCRough.rewards.scales):
            orientation = -5.0
            torques = -0.000025
            feet_air_time = 2.0  # feet_contact_forces = -0.01

    class commands(ConfigAnymalCRough.commands):
        yaw_from_heading_target = False
        resampling_time = 4.0

        class ranges(ConfigAnymalCRough.commands.ranges):
            ang_vel_yaw = [-1.5, 1.5]

    class domain_rand(ConfigAnymalCRough.domain_rands):
        class rigid_shape(ConfigAnymalCRough.domain_rands.rigid_shape):
            friction_range = [
                0.0,
                1.5,
            ]  # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.
