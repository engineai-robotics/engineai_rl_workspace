from engineai_gym.envs.base.rewards.rewards_base import RewardsBase
import torch


class RewardsTypeBaseVel(RewardsBase):
    def reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.env.base_lin_vel[:, 2])

    def reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)

    def reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.env.cfg.rewards.params.tracking_sigma)

    def reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(
            self.env.commands[:, 2] - self.env.base_ang_vel[:, 2]
        )
        return torch.exp(-ang_vel_error / self.env.cfg.rewards.params.tracking_sigma)

    def reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(
            torch.abs(self.env.dof_pos - self.env.default_dof_pos), dim=1
        ) * (torch.norm(self.env.commands[:, :2], dim=1) < 0.1)

    def reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.env.last_root_vel - self.env.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew

    def reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities.
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.env.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.env.base_ang_vel[:, :2], dim=1) * 5.0)

        c_update = (lin_mismatch + ang_mismatch) / 2.0

        return c_update

    def reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.env.commands[:, :2] - self.env.base_lin_vel[:, :2], dim=1
        )
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2.0 - linear_error

    def reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed.
        This function checks if the robot is moving too slow, too fast, or at the desired speed,
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.env.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.env.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(self.env.base_lin_vel[:, 0]) != torch.sign(
            self.env.commands[:, 0]
        )

        # Initialize reward tensor
        reward = torch.zeros_like(self.env.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.0
        # Speed within desired range
        reward[speed_desired] = 2.0
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.env.commands[:, 0].abs() > 0.1)
