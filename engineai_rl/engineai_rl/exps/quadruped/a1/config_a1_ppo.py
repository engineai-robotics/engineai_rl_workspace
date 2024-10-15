from engineai_rl.algos.ppo.config_ppo import ConfigPpo


class ConfigA1Ppo(ConfigPpo):
    class params(ConfigPpo.params):
        entropy_coef = 0.01
