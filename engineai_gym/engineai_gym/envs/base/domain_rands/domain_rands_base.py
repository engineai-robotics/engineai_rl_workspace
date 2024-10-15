from abc import ABC


class DomainRandsBase(ABC):
    def __init__(self, env):
        self.env = env

    def init_domain_rands(self):
        pass

    def init_buffer_on_create_env(self):
        pass

    def init_buffer_after_create_env(self):
        pass

    def init_rand_vec_on_create_env(self):
        pass

    def init_rand_vec_on_reset_idx(self, env_ids):
        pass

    def init_rand_vec_on_step(self):
        pass

    def init_rand_vec_on_decimation(self):
        pass

    def process_on_create_env(self, prop, env_ids):
        pass

    def process_on_reset_idx(self, env_ids):
        pass

    def process_on_step(self):
        pass

    def process_after_step(self):
        pass

    def process_on_decimation(self):
        pass

    def process_after_decimation(self):
        pass
