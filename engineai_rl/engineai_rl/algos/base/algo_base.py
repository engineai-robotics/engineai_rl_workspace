from abc import ABC


class AlgoBase(ABC):
    def __init__(self, networks, policy_cfg, env, device="cpu", **kwargs):
        self.device = device
        self.networks = networks

    def eval_mode(self):
        raise NotImplementedError

    def train_mode(self):
        raise NotImplementedError

    @property
    def inference_policy(self):
        raise NotImplementedError

    def act(self, inputs):
        raise NotImplementedError

    def process_env_step(self, rewards, dones, infos, **kwargs):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError
