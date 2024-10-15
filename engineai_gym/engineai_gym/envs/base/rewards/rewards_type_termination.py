from engineai_gym.envs.base.rewards.rewards_base import RewardsBase
import torch


class RewardsTypeTermination(RewardsBase):
    def reward_termination(self):
        # Terminal reward / penalty
        return self.env.reset_buf * ~self.env.time_out_buf
