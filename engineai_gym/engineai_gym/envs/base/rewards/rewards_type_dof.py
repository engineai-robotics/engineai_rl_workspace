from engineai_gym.envs.base.rewards.rewards_base import RewardsBase
import torch


class RewardsTypeDof(RewardsBase):
    def reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.env.torques), dim=1)

    def reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.env.dof_vel), dim=1)

    def reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(
            torch.square((self.env.last_dof_vel - self.env.dof_vel) / self.env.dt),
            dim=1,
        )

    def reward_dof_pos_limits(self):
        # soft limits
        m = (self.env.dof_pos_limits[:, 0] + self.env.dof_pos_limits[:, 1]) / 2
        r = self.env.dof_pos_limits[:, 1] - self.env.dof_pos_limits[:, 0]
        soft_dof_pos_lower_limits = (
            m - 0.5 * r * self.env.cfg.rewards.params.soft_dof_pos_limit_multi
        )
        soft_dof_pos_upper_limits = (
            m + 0.5 * r * self.env.cfg.rewards.params.soft_dof_pos_limit_multi
        )
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.env.dof_pos - soft_dof_pos_lower_limits).clip(
            max=0.0
        )  # lower limit
        out_of_limits += (self.env.dof_pos - soft_dof_pos_upper_limits).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (
                torch.abs(self.env.dof_vel)
                - self.env.dof_vel_limits
                * self.env.cfg.rewards.params.soft_dof_vel_limit_multi
            ).clip(min=0.0, max=1.0),
            dim=1,
        )

    def reward_torque_limits(self):
        # penalize torques too close to the limit
        soft_torque_limit = torch.ones_like(self.env.torque_limits)
        if self.env.cfg.rewards.params.soft_torque_limit_multi is not None:
            for i in range(self.env.num_dofs):
                for dof_type in self.env.cfg.rewards.params.soft_torque_limit_multi:
                    if dof_type in self.env.dof_names[i]:
                        soft_torque_limit[
                            i
                        ] = self.env.cfg.rewards.params.soft_torque_limit_multi[
                            dof_type
                        ]
        return torch.sum(
            (
                torch.abs(self.env.torques) - self.env.torque_limits * soft_torque_limit
            ).clip(min=0.0),
            dim=1,
        )

    def reward_dof_ref_pos_diff(self):
        """
        Calculates the reward based on the difference between the current joint positions and the reference joint positions.
        """
        diff = self.env.dof_pos - self.env.ref_dof_pos
        return torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(
            diff, dim=1
        ).clamp(0, 0.5)

    def reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.env.dof_pos - self.env.default_dof_pos
        left_yaw_roll = joint_diff[:, :2]
        right_yaw_roll = joint_diff[:, 6:8]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)
