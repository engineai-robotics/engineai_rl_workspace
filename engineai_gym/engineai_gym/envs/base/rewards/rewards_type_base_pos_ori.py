from engineai_gym.envs.base.rewards.rewards_base import RewardsBase
import torch


class RewardsTypeBasePosOri(RewardsBase):
    def reward_orientation(self):
        # Penalize non-flat base orientation
        return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)

    def reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(
            self.env.root_states[:, 2].unsqueeze(1) - self.env.measured_heights, dim=1
        )
        return torch.square(
            base_height - self.env.cfg.rewards.params.base_height_target
        )
