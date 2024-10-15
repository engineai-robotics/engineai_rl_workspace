from engineai_rl_lib.class_operations import (
    get_classes_of_base_from_files_with_prefix_in_folder,
    class_name_to_instance_name,
)
import os
from engineai_gym.envs.base.domain_rands import *


class DomainRands:
    def __init__(self, env):
        domain_rands_classes = get_classes_of_base_from_files_with_prefix_in_folder(
            os.path.dirname(__file__), "domain_rands_type", DomainRandsBase
        )

        for domain_rands_class in domain_rands_classes:
            instance_name = class_name_to_instance_name(domain_rands_class.__name__)
            exec(f"self.{instance_name} = domain_rands_class(env)")

    def init_domain_rands(self):
        self.instances = [
            getattr(self, attr)
            for attr in dir(self)
            if attr.startswith("domain_rands_type_")
        ]
        for instance in self.instances:
            instance.init_domain_rands()

    def init_buffer_on_create_env(self):
        for instance in self.instances:
            instance.init_buffer_on_create_env()

    def init_buffer_after_create_env(self):
        for instance in self.instances:
            instance.init_buffer_after_create_env()

    def init_rand_vec_on_create_env(self):
        for instance in self.instances:
            instance.init_rand_vec_on_create_env()

    def init_rand_vec_on_reset_idx(self, env_ids):
        for instance in self.instances:
            instance.init_rand_vec_on_reset_idx(env_ids)

    def init_rand_vec_on_step(self):
        for instance in self.instances:
            instance.init_rand_vec_on_step()

    def init_rand_vec_on_decimation(self):
        for instance in self.instances:
            instance.init_rand_vec_on_decimation()

    def process_on_reset_idx(self, env_ids):
        for instance in self.instances:
            instance.process_on_reset_idx(env_ids)

    def process_on_decimation(self):
        for instance in self.instances:
            instance.process_on_decimation()

    def process_after_decimation(self):
        for instance in self.instances:
            instance.process_after_decimation()

    def process_after_step(self):
        for instance in self.instances:
            instance.process_after_step()
