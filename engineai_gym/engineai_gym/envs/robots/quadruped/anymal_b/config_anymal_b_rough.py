from engineai_gym.envs.robots.quadruped.anymal_c.rough.config_anymal_c_rough import (
    ConfigAnymalCRough,
)


class ConfigAnymalBRough(ConfigAnymalCRough):
    class asset(ConfigAnymalCRough.asset):
        file = "{ENGINEAI_GYM_PACKAGE_DIR}/resources/robots/quadruped/anymal_b/urdf/anymal_b.urdf"
        name = "anymal_b"
        foot_name = "FOOT"

    class rewards(ConfigAnymalCRough.rewards):
        class scales(ConfigAnymalCRough.rewards.scales):
            pass
