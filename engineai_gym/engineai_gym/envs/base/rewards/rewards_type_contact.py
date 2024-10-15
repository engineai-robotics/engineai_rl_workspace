from engineai_gym.envs.base.rewards.rewards_base import RewardsBase
import torch


class RewardsTypeContact(RewardsBase):
    def reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum(
            (
                torch.norm(self.env.contact_forces[:, self.env.foot_indices, :], dim=-1)
                - self.env.cfg.rewards.params.max_contact_force
            ).clip(min=0.0),
            dim=1,
        )

    def reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        foot_speed_norm = torch.norm(
            self.env.rigid_body_state[:, self.env.foot_indices, 7:9], dim=2
        )
        rew = torch.sqrt(foot_speed_norm)
        rew *= self.env.contact
        return torch.sum(rew, dim=1)

    def reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase.
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        stance_mask, _, _, _ = self.env.get_gait_phase()
        reward = torch.where(self.env.contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)
