from engineai_gym.envs.base.rewards.rewards_base import RewardsBase
import torch


class RewardsTypeGait(RewardsBase):
    def reward_feet_air_time(self):
        # Reward long steps
        rew_airTime = torch.sum(
            (self.env.foot_air_time - 0.5) * self.env.first_contact, dim=1
        )  # reward only on first contact with the ground
        rew_airTime *= (
            torch.norm(self.env.commands[:, :2], dim=1) > 0.1
        )  # no reward for zero command
        return rew_airTime

    def reward_no_fly(self):
        contacts = self.env.contact_forces[:, self.env.foot_indices, 2] > 0.1
        single_contact = torch.sum(1.0 * contacts, dim=1) == 1
        return 1.0 * single_contact

    def reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(
            torch.norm(self.env.contact_forces[:, self.env.foot_indices, :2], dim=2)
            > 5 * torch.abs(self.env.contact_forces[:, self.env.foot_indices, 2]),
            dim=1,
        )

    def reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        if self.env.foot_indices.numel() == 0:
            raise RuntimeError("Feet are not specified!")
        foot_pos = self.env.rigid_body_state[:, self.env.foot_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.env.cfg.rewards.params.min_feet_dist
        max_df = self.env.cfg.rewards.params.max_feet_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (
            torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)
        ) / 2

    def reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        if self.env.knee_indices.numel() == 0:
            raise RuntimeError("Knees are not specified!")
        knee_pos = self.env.rigid_body_state[:, self.env.knee_indices, :2]
        knee_dist = torch.norm(knee_pos[:, 0, :] - knee_pos[:, 1, :], dim=1)
        fd = self.env.cfg.rewards.params.min_feet_dist
        max_df = self.env.cfg.rewards.params.max_feet_dist / 2
        d_min = torch.clamp(knee_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(knee_dist - max_df, 0, 0.5)
        return (
            torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)
        ) / 2

    def reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """

        # Compute swing mask
        _, swing_mask, _, swing_curve = self.env.get_gait_phase()

        # feet height should larger than target feet height at the peak
        rew_pos = torch.norm(
            (
                swing_curve * self.env.cfg.rewards.params.target_feet_height
                - self.env.feet_heights
            )
            * swing_mask,
            dim=1,
        )
        return rew_pos

    def reward_feet_height(self):
        foot_velocities = self.env.rigid_body_state[:, self.env.foot_indices, 10:13]
        reward = torch.sum(
            torch.exp(
                -self.env.feet_heights / self.env.cfg.rewards.params.target_feet_height
            )
            * torch.exp(
                -torch.norm(self.env.commands[:, :3], dim=1, keepdim=True)
            ).repeat(1, len(self.env.foot_indices)),
            dim=1,
        )

        reward += torch.sum(
            torch.exp(
                -self.env.feet_heights / self.env.cfg.rewards.params.target_feet_height
            )
            * torch.square(torch.norm(foot_velocities[:, :, :2], dim=-1)),
            dim=1,
        )
        reward += torch.sum(
            torch.exp(
                -self.env.feet_heights
                / (self.env.cfg.rewards.params.target_feet_height * 0.5)
            )
            * torch.square(torch.abs(foot_velocities[:, :, 2])),
            dim=1,
        )
        return reward
