from engineai_gym.envs.base.rewards.rewards_base import RewardsBase
import torch


class RewardsTypeAction(RewardsBase):
    def reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_actions - self.env.actions), dim=1)

    def reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(
            torch.square(self.env.last_actions - self.env.actions), dim=1
        )
        term_2 = torch.sum(
            torch.square(
                self.env.actions
                + self.env.last_last_actions
                - 2 * self.env.last_actions
            ),
            dim=1,
        )
        term_3 = 0.05 * torch.sum(torch.abs(self.env.actions), dim=1)
        return term_1 + term_2 + term_3
