from engineai_gym.envs.base.rewards.rewards_base import RewardsBase
from engineai_rl_lib.class_operations import (
    get_classes_of_base_from_files_with_prefix_in_folder,
    class_name_to_instance_name,
)
import os


class Rewards(RewardsBase):
    def __init__(self, env):
        super().__init__(env)
        self._get_all_rewards()

    def _get_all_rewards(self):
        rewards_classes = get_classes_of_base_from_files_with_prefix_in_folder(
            os.path.dirname(__file__), "rewards_type", RewardsBase
        )
        for rewards_class in rewards_classes:
            instance_name = class_name_to_instance_name(rewards_class.__name__)
            exec(f"self.{instance_name} = rewards_class(self.env)")
            for method_name in dir(eval(f"self.{instance_name}")):
                if method_name.startswith("__"):
                    continue  # skip Python internal methods
                method = getattr(eval(f"self.{instance_name}"), method_name)
                if callable(method) and not hasattr(self, method_name):
                    setattr(self, method_name, method)
