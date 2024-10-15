import torch
from isaacgym.torch_utils import quat_rotate_inverse


class Obs:
    def __init__(self, env):
        self.env = env

    def base_z_pos(self):
        return self.env.root_states[:, 2:3]

    def base_lin_vel(self):
        return self.env.base_lin_vel

    def base_ang_vel(self, lag=False):
        if lag:
            try:
                return (
                    self.env.domain_rands.domain_rands_type_obs_lag.lagged_base_ang_vel
                )
            except:
                return None
        else:
            return self.env.base_ang_vel

    def projected_gravity(self):
        return self.env.projected_gravity

    def dof_pos(self, lag=False):
        if lag:
            try:
                return (
                    self.env.domain_rands.domain_rands_type_obs_lag.lagged_dof_pos
                    - self.env.default_dof_pos
                )
            except:
                return None
        else:
            return self.env.dof_pos - self.env.default_dof_pos

    def absolute_dof_pos(self, lag=False):
        if lag:
            try:
                return self.env.domain_rands.domain_rands_type_obs_lag.lagged_dof_pos
            except:
                return None
        else:
            return self.env.dof_pos

    def dof_vel(self, lag=False):
        if lag:
            try:
                return self.env.domain_rands.domain_rands_type_obs_lag.lagged_dof_vel
            except:
                return None
        else:
            return self.env.dof_vel

    def actions(self, lag=False):
        if lag:
            try:
                return self.env.domain_rands.domain_rands_type_action_lag.lagged_actions
            except:
                return None
        else:
            return self.env.actions

    def height_measurements(self):
        if self.env.cfg.terrain.measure_heights:
            return torch.clip(
                self.env.root_states[:, 2].unsqueeze(1)
                - 0.5
                - self.env.measured_heights,
                -1,
                1.0,
            )
        else:
            raise RuntimeError(
                "measure_heights is false so height_measurements obs can not be computed!"
            )

    def dof_pos_ref_diff(self):
        return self.env.dof_pos - self.env.ref_dof_pos

    def rand_push_force(self):
        return self.env.domain_rands.domain_rands_type_disturbance.rand_push_force[
            :, :2
        ]

    def rand_push_torque(self):
        return self.env.domain_rands.domain_rands_type_disturbance.rand_push_torque

    def terrain_frictions(self):
        return self.env.env_frictions

    def body_mass(self):
        return self.env.body_mass

    def stance_curve(self):
        _, _, stance_curve, _ = self.env.get_gait_phase()
        return stance_curve

    def swing_curve(self):
        _, _, _, swing_curve = self.env.get_gait_phase()
        return swing_curve

    def contact_mask(self):
        return (
            self.env.contact_forces[:, self.env.foot_indices, 2]
            > self.env.cfg.contact.foot_contact_threshold
        )

    def base_euler_xyz(self, lag=False):
        if lag:
            try:
                return (
                    self.env.domain_rands.domain_rands_type_obs_lag.lagged_base_euler_xyz
                )
            except:
                return None
        else:
            return self.env.base_euler_xyz

    def foot_pos(self):
        if self.env.foot_indices.numel() == 0:
            raise RuntimeError("Feet are not specified!")
        foot_pos_list = []
        for idx in self.env.foot_indices:
            foot_pos_list.append(
                quat_rotate_inverse(
                    self.env.base_quat,
                    self.env.rigid_body_state[:, idx, :3] - self.env.root_states[:, :3],
                )
            )
        return torch.cat(foot_pos_list, dim=-1)
