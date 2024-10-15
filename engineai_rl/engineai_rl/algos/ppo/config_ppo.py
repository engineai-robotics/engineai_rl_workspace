from engineai_rl.algos.base.base_algo_config import ConfigAlgoBase


class ConfigPpo(ConfigAlgoBase):
    class params(ConfigAlgoBase.params):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 0.001
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
