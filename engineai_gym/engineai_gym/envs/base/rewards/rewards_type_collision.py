from engineai_gym.envs.base.rewards.rewards_base import RewardsBase
import torch


class RewardsTypeCollision(RewardsBase):
    def reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(
            1.0
            * (
                torch.norm(
                    self.env.contact_forces[:, self.env.penalised_contact_indices, :],
                    dim=-1,
                )
                > 0.1
            ),
            dim=1,
        )
