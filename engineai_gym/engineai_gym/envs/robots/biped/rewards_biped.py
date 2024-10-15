from engineai_gym.envs.base.rewards.rewards import Rewards
import torch


class RewardsBiped(Rewards):
    def reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(
            -torch.sum(torch.abs(self.env.base_euler_xyz[:, :2]), dim=1) * 10
        )
        orientation = torch.exp(
            -torch.norm(self.env.projected_gravity[:, :2], dim=1) * 20
        )
        return (quat_mismatch + orientation) / 2.0

    def reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(
            torch.abs(self.env.dof_pos - self.env.default_dof_pos), dim=1
        ) * (torch.norm(self.env.commands[:, :2], dim=1) < 0.15)

    def reward_feet_air_time(self):
        # Reward long steps
        air_time = self.env.foot_air_time.clamp(0, 0.5) * self.env.first_contact
        return air_time.sum(dim=1)

    def reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum(
            (
                torch.norm(self.env.contact_forces[:, self.env.foot_indices, :], dim=-1)
                - self.env.cfg.rewards.params.max_contact_force
            ).clip(0, 350),
            dim=1,
        )

    def reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height
        of its feet when they are in contact with the ground.
        """
        stance_mask, _, _, _ = self.env.get_gait_phase()
        measured_heights = torch.sum(
            self.env.rigid_body_state[:, self.env.foot_indices, 2] * stance_mask, dim=1
        ) / torch.sum(stance_mask, dim=1)
        base_height = self.env.root_states[:, 2] - (measured_heights - 0.05)
        return torch.exp(
            -torch.abs(base_height - self.env.cfg.rewards.params.base_height_target)
            * 100
        )
