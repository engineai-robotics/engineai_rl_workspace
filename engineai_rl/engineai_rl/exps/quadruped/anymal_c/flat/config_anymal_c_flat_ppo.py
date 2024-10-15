from engineai_rl.algos.ppo.config_ppo import ConfigPpo


class ConfigAnymalCFlatPpo(ConfigPpo):
    class policy(ConfigPpo.policy):
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class params(ConfigPpo.params):
        entropy_coef = 0.01

    class runner(ConfigPpo.runner):
        max_iterations = 300
